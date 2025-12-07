import os
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

MAX_LEN = 1024
STAY_LIMIT = 300
SHARD_SIZE = 2000
class LabTOPHiRIDPreprocessor:
    def __init__(
        self,
        data_dir_1,
        data_dir_2,
        model_name="gpt2",
        max_len=MAX_LEN,
        stay_limit=STAY_LIMIT,
        shard_size=SHARD_SIZE,
        out_dir="stay_limit_300_gpt_new_processed_data",
    ):
        """
        HiRID -> LabTOP-style preprocessor (Option A: with event-type tokens).

        data_dir_1: HiRID raw_stage root, e.g. .../hirid/1.1.1/raw_stage
        data_dir_2: HiRID reference_data root, e.g. .../hirid/1.1.1/reference_data

        Key design choices:
        - Time window: first 72h of ICU stay
        - LOS filter: keep stays with LOS >= 6h (approx from last event time)
        - Event types: labevent (observation_tables) / inputevent (pharma_records)
        - Item names: Variable Name from hirid_variable_reference where possible
        - Units: Unit from variable reference for labs, doseunit for pharmas
        - Value formatting: LabTOP-style char-level "1 1 9 . 0"
        - type_ids: 1 on numeric characters only, 0 elsewhere
        - Eval items: prompt = history + time + event_type + item_name,
                      label = value + unit + [EOE]
        """

        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.max_len = max_len
        self.stay_limit = stay_limit
        self.shard_size = shard_size
        self.out_dir = out_dir

        # --- tokenizer (LabTOP pattern) ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # time tokens + event-type tokens + [EOE]
        self.days = [f"[DAY{i}]" for i in range(4)]
        self.weekdays = ["[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"]
        self.hours = [f"[{i:02d}h]" for i in range(24)]
        self.minutes = [f"[{i:02d}m]" for i in range(60)]
        self.event_types = ["labevent", "inputevent"]
        self.other_specials = ["[EOE]"]

        new_tokens = (
            self.days
            + self.weekdays
            + self.hours
            + self.minutes
            + self.event_types
            + self.other_specials
        )
        self.tokenizer.add_tokens(new_tokens)

        os.makedirs(self.out_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.out_dir)

        self.time_cache = {}

        # variable reference mappings
        self.lab_name = {}
        self.lab_unit = {}
        self.pharm_name = {}
        self.pharm_unit = {}
        self._load_variable_reference()

    # ------------------ variable reference ------------------

    def _load_variable_reference(self):
        """
        Load hirid_variable_reference.csv with columns:
        'Source Table', 'ID', 'Variable Name', 'Unit', 'Additional information'
        and build separate maps for Observation vs Pharma.
        """
        ref_path = os.path.join(self.data_dir_2, "hirid_variable_reference.csv")
        if not os.path.exists(ref_path):
            print(f"WARNING: variable reference not found at {ref_path}")
            return

        print(f"Loading variable reference from {ref_path}...")
        ref = pd.read_csv(ref_path)

        # Normalize column names
        cols = {c.lower().strip(): c for c in ref.columns}
        src_col = cols.get("source table".lower(), "Source Table")
        id_col = cols.get("id", "ID")
        name_col = cols.get("variable name".lower(), "Variable Name")
        unit_col = cols.get("unit", "Unit")

        # Make sure IDs are ints when possible
        ref[id_col] = pd.to_numeric(ref[id_col], errors="coerce").astype("Int64")

        # Observation variables (labs, vitals, etc.)
        obs = ref[ref[src_col] == "Observation"].dropna(subset=[id_col])
        self.lab_name = dict(
            zip(obs[id_col].astype(int), obs[name_col].astype(str).str.lower())
        )
        self.lab_unit = dict(
            zip(obs[id_col].astype(int), obs[unit_col].fillna("").astype(str).str.lower())
        )

        # Pharma variables (drugs)
        pharm = ref[ref[src_col] == "Pharma"].dropna(subset=[id_col])
        self.pharm_name = dict(
            zip(pharm[id_col].astype(int), pharm[name_col].astype(str).str.lower())
        )
        self.pharm_unit = dict(
            zip(pharm[id_col].astype(int), pharm[unit_col].fillna("").astype(str).str.lower())
        )

        print(
            f"Variable reference loaded: {len(self.lab_name)} observation vars, "
            f"{len(self.pharm_name)} pharma vars."
        )

    # ------------------ general table ------------------

    def load_general(self):
        gen_path = os.path.join(self.data_dir_2, "general_table.csv")
        print(f"Loading general_table from {gen_path}...")
        df = pd.read_csv(gen_path)

        df["admissiontime"] = pd.to_datetime(df["admissiontime"])

        df = df.rename(
            columns={
                "patientid": "stay_id",
                "admissiontime": "intime",
                "sex": "gender",
                "age": "anchor_age",
            }
        )

        if self.stay_limit:
            df = df.head(self.stay_limit)

        return df

    # ------------------ data readers (parquet or csv) ------------------

    def _get_obs_paths(self):
        base = os.path.join(self.data_dir_1, "observation_tables")
        parquet_dir = os.path.join(base, "parquet")
        csv_dir = os.path.join(base, "csv")

        if os.path.isdir(parquet_dir):
            return sorted(glob.glob(os.path.join(parquet_dir, "part-*")))
        elif os.path.isdir(csv_dir):
            return sorted(glob.glob(os.path.join(csv_dir, "part-*")))
        else:
            return []

    def _get_pharma_paths(self):
        base = os.path.join(self.data_dir_1, "pharma_records")
        parquet_dir = os.path.join(base, "parquet")
        csv_dir = os.path.join(base, "csv")

        if os.path.isdir(parquet_dir):
            return sorted(glob.glob(os.path.join(parquet_dir, "part-*")))
        elif os.path.isdir(csv_dir):
            return sorted(glob.glob(os.path.join(csv_dir, "part-*")))
        else:
            return []

    def _read_part(self, path, columns):
        """
        Read a single part file (parquet or csv) with the given columns.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in [".parquet", ".pq"]:
            return pd.read_parquet(path, columns=columns)
        else:
            # assume CSV
            return pd.read_csv(path, usecols=columns)

    def load_observations_chunked(self, stay_ids_set):
        """
        Load ALL observation_tables rows for the selected stays.
        No type filtering, to mirror HiRID config where 'lab_table' = observation_tables.
        """
        paths = self._get_obs_paths()
        print(f"Loading observations from {len(paths)} partitions...")

        chunks = []
        for path in paths:
            print(f"Reading {path}...")
            cols = ["patientid", "datetime", "value", "variableid"]
            df = self._read_part(path, columns=cols)
            df = df[df["patientid"].isin(stay_ids_set)]
            if df.empty:
                continue

            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.rename(
                columns={
                    "patientid": "stay_id",
                    "datetime": "time",
                    "value": "value",
                    "variableid": "itemid",
                }
            )
            chunks.append(df)

        if not chunks:
            return pd.DataFrame(columns=["stay_id", "time", "itemid", "value"])

        return pd.concat(chunks, ignore_index=True)

    def load_pharma_chunked(self, stay_ids_set):
        paths = self._get_pharma_paths()
        print(f"Loading pharma_records from {len(paths)} partitions...")

        chunks = []
        for path in paths:
            print(f"Reading {path}...")
            cols = ["patientid", "pharmaid", "givenat", "givendose", "doseunit"]
            df = self._read_part(path, columns=cols)
            df = df[df["patientid"].isin(stay_ids_set)]
            if df.empty:
                continue

            df["givenat"] = pd.to_datetime(df["givenat"])
            df = df.rename(
                columns={
                    "patientid": "stay_id",
                    "pharmaid": "itemid",
                    "givenat": "time",
                    "givendose": "value",
                    "doseunit": "uom",
                }
            )
            chunks.append(df)

        if not chunks:
            return pd.DataFrame(columns=["stay_id", "time", "itemid", "value", "uom"])

        return pd.concat(chunks, ignore_index=True)

    # ------------------ helpers ------------------

    def format_value(self, v):
        """
        LabTOP-style: char-level digits, no rounding.
        NaNs are dropped.
        """
        if pd.isna(v):
            return None
        s = str(v)
        return " ".join(list(s))

    def get_time_tokens(self, dt, intime):
        """
        [DAYd] [WEEKDAY] [HHh] [MMm] with caching.
        """
        delta = dt - intime
        hours_total = int(delta.total_seconds() // 3600)
        minutes_total = int(delta.total_seconds() // 60)

        if hours_total < 0:
            hours_total = 0
            minutes_total = 0

        day = min(hours_total // 24, 3)
        hour = hours_total % 24
        minute = minutes_total % 60
        weekday_idx = dt.weekday()

        key = (day, weekday_idx, hour, minute)
        if key in self.time_cache:
            return self.time_cache[key]

        tok = f"[DAY{day}] {self.weekdays[weekday_idx]} [{hour:02d}h] [{minute:02d}m]"
        self.time_cache[key] = tok
        return tok

    def _item_name(self, event_type, itemid):
        """
        Human-readable item name when available, else fallback.
        """
        try:
            iid = int(itemid)
        except Exception:
            iid = itemid

        if event_type == "labevent" and iid in self.lab_name:
            return self.lab_name[iid]
        if event_type == "inputevent" and iid in self.pharm_name:
            return self.pharm_name[iid]
        return f"{event_type}_item_{iid}"

    def _item_unit(self, event_type, itemid, raw_uom):
        """
        Prefer variable_reference Unit for labs; for pharmas, use doseunit
        if present, else fallback to reference Unit.
        """
        try:
            iid = int(itemid)
        except Exception:
            iid = itemid

        # Normalize raw_uom
        if isinstance(raw_uom, str):
            ru = raw_uom.strip().lower()
        else:
            ru = ""

        if event_type == "labevent":
            if iid in self.lab_unit and self.lab_unit[iid]:
                return self.lab_unit[iid]
            return ru  # just in case

        # inputevent (pharma)
        if ru:
            return ru
        if iid in self.pharm_unit and self.pharm_unit[iid]:
            return self.pharm_unit[iid]
        return ""

    # ------------------ main processing ------------------

    def process_data(self):
        icu_df = self.load_general()
        if icu_df.empty:
            print("No stays in general_table.")
            return []

        stay_ids = set(icu_df["stay_id"].unique())

        obs_df = self.load_observations_chunked(stay_ids)
        pharm_df = self.load_pharma_chunked(stay_ids)

        if obs_df.empty and pharm_df.empty:
            print("No events in obs/pharma for selected stays.")
            return []

        obs_df["event_type"] = "labevent"
        pharm_df["event_type"] = "inputevent"

        # Ensure uom columns exist
        if "uom" not in obs_df.columns:
            obs_df["uom"] = ""

        all_df = pd.concat([obs_df, pharm_df], ignore_index=True)
        all_df = all_df.sort_values(["stay_id", "time"])

        # LOS >= 6h based on last event time
        last_times = all_df.groupby("stay_id")["time"].max()
        icu_df = icu_df.merge(last_times.rename("last_time"), on="stay_id", how="left")
        icu_df = icu_df.dropna(subset=["last_time"])

        icu_df["los_hours"] = (
            (icu_df["last_time"] - icu_df["intime"]).dt.total_seconds() / 3600.0
        )
        icu_df = icu_df[icu_df["los_hours"] >= 6]

        print(f"Stays after LOS >= 6h filter: {len(icu_df)}")

        grouped = all_df.groupby("stay_id")
        data = []

        for _, row in tqdm(icu_df.iterrows(), total=len(icu_df), desc="Building stay event streams"):
            sid = row["stay_id"]
            intime = row["intime"]
            cutoff = intime + pd.Timedelta(hours=24)

            if sid not in grouped.groups:
                continue

            ev = grouped.get_group(sid)
            ev = ev[(ev["time"] >= intime) & (ev["time"] <= cutoff)]
            if ev.empty:
                continue

            data.append(
                {
                    "stay_id": sid,
                    "demographics": row,
                    "events": ev,
                    "intime": intime,
                }
            )

        return data

    # ------------------ sequence creation ------------------

    def create_sequences(self, data, split):
        sequences = []
        eval_items = []
        shard_idx = 0

        for entry in tqdm(data, desc=f"Creating sequences ({split})"):
            sid = entry["stay_id"]
            demo = entry["demographics"]
            events = entry["events"]
            intime = entry["intime"]

            gender = str(demo["gender"]).lower()
            age = str(demo["anchor_age"])
            status = str(demo.get("discharge_status", "")).lower()

            demo_text = f"gender {gender} age {age} status {status}"
            demo_tok = self.tokenizer.encode(demo_text, add_special_tokens=False)

            curr_ids = demo_tok.copy()
            curr_types = [0] * len(demo_tok)

            for _, e in events.iterrows():
                t_str = self.get_time_tokens(e["time"], intime)
                t_tok = self.tokenizer.encode(t_str, add_special_tokens=False)

                etype = e["event_type"]
                etype_id = self.tokenizer.convert_tokens_to_ids(etype)

                item_name = self._item_name(etype, e["itemid"])
                item_tok = self.tokenizer.encode(item_name, add_special_tokens=False)

                v_str = self.format_value(e["value"])
                if v_str is None:
                    continue
                v_tok = self.tokenizer.encode(v_str, add_special_tokens=False)

                uom_text = self._item_unit(etype, e["itemid"], e.get("uom", ""))
                if uom_text:
                    uom_tok = self.tokenizer.encode(uom_text + " [EOE]", add_special_tokens=False)
                else:
                    uom_tok = self.tokenizer.encode("[EOE]", add_special_tokens=False)

                event_toks = t_tok + [etype_id] + item_tok + v_tok + uom_tok

                type_ids = (
                    [0] * len(t_tok)      # time tokens
                    + [0]                 # event-type token
                    + [0] * len(item_tok) # item name
                    + [1] * len(v_tok)    # numeric chars
                    + [0] * len(uom_tok)  # unit + [EOE]
                )

                # if adding this event would overflow, start a new sequence with fresh demo prefix
                if len(curr_ids) + len(event_toks) > self.max_len:
                    sequences.append(
                        {"stay_id": sid, "input_ids": curr_ids, "type_ids": curr_types}
                    )
                    curr_ids = demo_tok.copy()
                    curr_types = [0] * len(demo_tok)

                # eval items for val/test: prompt ends at item name
                if split in ["val", "test"]:
                    prompt = curr_ids + t_tok + [etype_id] + item_tok
                    if len(prompt) > self.max_len:
                        tail = prompt[-(self.max_len - len(demo_tok)):]
                        prompt = demo_tok + tail

                    eval_items.append(
                        {
                            "stay_id": sid,
                            "prompt_ids": prompt,
                            "label_ids": v_tok + uom_tok,
                            "valuenum": e["value"],
                            "itemid": e["itemid"],
                            "event_type": etype,
                        }
                    )

                curr_ids.extend(event_toks)
                curr_types.extend(type_ids)

            if len(curr_ids) > len(demo_tok):
                sequences.append(
                    {"stay_id": sid, "input_ids": curr_ids, "type_ids": curr_types}
                )

            if len(sequences) >= self.shard_size:
                self._write_shard(sequences, split, shard_idx)
                sequences = []
                shard_idx += 1

        if sequences:
            self._write_shard(sequences, split, shard_idx)
            shard_idx += 1

        self._merge_shards(split, shard_idx)

        if eval_items:
            eval_path = os.path.join(self.out_dir, f"{split}_eval.pkl")
            with open(eval_path, "wb") as f:
                pickle.dump(eval_items, f)
            print(f"Saved {split}_eval.pkl with {len(eval_items)} items.")

    # ------------------ shard helpers ------------------

    def _write_shard(self, data, split, idx):
        fn = os.path.join(self.out_dir, f"{split}_shard_{idx}.pkl")
        with open(fn, "wb") as f:
            pickle.dump(data, f)

    def _merge_shards(self, split, n):
        all_data = []
        for i in range(n):
            fn = os.path.join(self.out_dir, f"{split}_shard_{i}.pkl")
            if os.path.exists(fn):
                with open(fn, "rb") as f:
                    all_data.extend(pickle.load(f))
                os.remove(fn)

        out_path = os.path.join(self.out_dir, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"Saved {split}.pkl with {len(all_data)} sequences.")

    # ------------------ run ------------------

    def run(self):
        data = self.process_data()
        if not data:
            print("No data after processing.")
            return

        stays = [d["stay_id"] for d in data]

        if len(stays) < 10:
            train, val, test = data, [], []
            print("Too few stays for 8:1:1 split; using all for train.")
        else:
            train_ids, tmp_ids = train_test_split(stays, test_size=0.3, random_state=42)
            val_ids, test_ids = train_test_split(tmp_ids, test_size=0.5, random_state=42)

            train = [d for d in data if d["stay_id"] in train_ids]
            val = [d for d in data if d["stay_id"] in val_ids]
            test = [d for d in data if d["stay_id"] in test_ids]

        print(
            f"HiRID train stays: {len(train)}, val: {len(val)}, test: {len(test)}"
        )

        self.create_sequences(train, "train")
        if val:
            self.create_sequences(val, "val")
        if test:
            self.create_sequences(test, "test")

        print("HiRID preprocessing complete.")


if __name__ == "__main__":
    data_dir_1 = "/Users/ziwei/physionet.org/files/hirid/1.1.1/raw_stage"
    data_dir_2 = "/Users/ziwei/physionet.org/files/hirid/1.1.1/reference_data"
    out_dir = "stay_limit_300_gpt_new_processed_data"   # <-- make this match your training data_dir

    os.makedirs(out_dir, exist_ok=True)
    pre = LabTOPHiRIDPreprocessor(
        data_dir_1=data_dir_1,
        data_dir_2=data_dir_2,
        max_len=MAX_LEN,
        stay_limit=STAY_LIMIT,
        shard_size=SHARD_SIZE,
        out_dir=out_dir,
    )
    pre.run()
