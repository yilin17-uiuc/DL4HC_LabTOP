import os
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

MAX_LEN = 2048
STAY_LIMIT = 10000
SHARD_SIZE = 2000


class LabTOPEICUPreprocessor:
    """
    eICU → LabTOP-style preprocessor (original LabTOP behavior, no vitals).

    Tables used (from eICU table_candidates):
      - lab           (supervised)
      - microLab      (context only)
      - intakeOutput  (context, numeric)
      - infusionDrug  (context, numeric)
      - medication    (context, numeric; dosage string parsed)
      - treatment     (context only)

    Key design:
      - Time encoding: [DAYd] [HHh] [MMm], no weekday.
      - DAY index: 0..507 inclusive (matches original eICU config).
      - Time base: offset (minutes since ICU admission), clipped at 0.
      - Window: first 24h of ICU stay.
      - LOS filter: keep stays with LOS >= 6h (based on last event).
      - 90% cumulative frequency filtering PER TABLE (on item_name).
      - type_ids: 1 only on numeric chars of labevent; 0 everywhere else.
      - Eval items (val/test): only labevents (labs) are targets.
    """

    def __init__(
        self,
        data_dir,
        model_name="gpt2",
        max_len=MAX_LEN,
        stay_limit=STAY_LIMIT,
        shard_size=SHARD_SIZE,
        out_dir="eicu_labtop_processed",
    ):
        self.data_dir = data_dir
        self.max_len = max_len
        self.stay_limit = stay_limit
        self.shard_size = shard_size
        self.out_dir = out_dir

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # time tokens (no weekday) — NOTE: 0..507 inclusive
        self.days = [f"[DAY{i}]" for i in range(508)]
        self.hours = [f"[{i:02d}h]" for i in range(24)]
        self.minutes = [f"[{i:02d}m]" for i in range(60)]

        # event-type tokens
        self.event_types = [
            "labevent",
            "microevent",
            "intakeevent",
            "infusionevent",
            "medevent",
            "treatmentevent",
        ]

        self.other_specials = ["[EOE]"]

        new_tokens = self.days + self.hours + self.minutes + self.event_types + self.other_specials
        self.tokenizer.add_tokens(new_tokens)

        os.makedirs(self.out_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.out_dir)

        self.time_cache = {}

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def format_value(self, v):
        """Char-level string: 123.4 -> '1 2 3 . 4'."""
        if pd.isna(v):
            return None
        s = str(v)
        return " ".join(list(s))

    def get_time_tokens(self, offset_min):
        """
        offset_min (minutes since ICU admit) -> [DAYd] [HHh] [MMm].
        DAY index: 1..507 inclusive for eICU.
        """
        if offset_min < 0:
            offset_min = 0

        hours_total = int(offset_min // 60)
        minute = int(offset_min % 60)

        # base day from offset, then shift to start at 1 and cap at 507
        raw_day = hours_total // 24          # 0,1,2,...
        day = max(1, min(raw_day + 1, 507))  # 1..507

        hour = hours_total % 24

        key = (day, hour, minute)
        if key in self.time_cache:
            return self.time_cache[key]

        tok = f"[DAY{day}] [{hour:02d}h] [{minute:02d}m]"
        self.time_cache[key] = tok
        return tok


    def _item_name(self, etype, name):
        return str(name).lower()

    def _item_unit(self, etype, uom):
        if isinstance(uom, str) and uom.strip():
            return uom.strip().lower()
        return ""

    def _empty_df(self):
        cols = ["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]
        return pd.DataFrame(columns=cols)

    # simple dosage parser: "10 mg" -> (10.0, "mg")
    # "1.5 units/hour" -> (1.5, "units/hour")
    def _parse_dosage(self, s):
        if not isinstance(s, str):
            return None, ""
        s = s.strip()
        if not s:
            return None, ""
        parts = s.split()
        try:
            val = float(parts[0])
        except Exception:
            return None, ""
        unit = " ".join(parts[1:]).strip()
        return val, unit

    # --------------------------------------------------------
    # Patient
    # --------------------------------------------------------

    def load_patient(self):
        path = os.path.join(self.data_dir, "patient.csv.gz")
        print(f"Loading patient table from {path}...")
        df = pd.read_csv(path, compression="gzip")

        df = df.rename(columns={"patientunitstayid": "stay_id"})

        # admission time
        if "unitadmittime24" in df.columns:
            time_col = "unitadmittime24"
        elif "unitadmissiontime24" in df.columns:
            time_col = "unitadmissiontime24"
        else:
            time_col = "hospitaladmittime24"

        df["intime"] = pd.to_datetime(df[time_col])

        # gender
        if "gender" in df.columns:
            df["gender"] = df["gender"]
        elif "sex" in df.columns:
            df["gender"] = df["sex"]
        else:
            df["gender"] = "unknown"

        # age
        if "age" in df.columns:
            df["anchor_age"] = df["age"]
        else:
            df["anchor_age"] = -1

        # discharge status
        if "unitdischargestatus" in df.columns:
            df["discharge_status"] = df["unitdischargestatus"]
        elif "hospitaldischargestatus" in df.columns:
            df["discharge_status"] = df["hospitaldischargestatus"]
        else:
            df["discharge_status"] = ""

        cols = ["stay_id", "intime", "gender", "anchor_age", "discharge_status"]
        df = df[cols].dropna(subset=["intime"])

        if self.stay_limit:
            df = df.head(self.stay_limit)

        print(f"Loaded {len(df)} stays from patient table.")
        return df

    # --------------------------------------------------------
    # Table loaders
    # --------------------------------------------------------

    def load_labs(self, stay_ids):
        path = os.path.join(self.data_dir, "lab.csv.gz")
        if not os.path.exists(path):
            print(f"WARNING: lab file not found at {path}")
            return self._empty_df()

        print(f"Loading lab from {path}...")
        df = pd.read_csv(path, compression="gzip")
        df = df[df["patientunitstayid"].isin(stay_ids)]
        if df.empty:
            print("lab: no rows for selected stays.")
            return self._empty_df()

        df = df.rename(
            columns={
                "patientunitstayid": "stay_id",
                "labresultoffset": "offset_min",
                "labname": "item_name",
                "labresult": "value",
            }
        )

        # prefer revised offset if available
        if "labresultrevisedoffset" in df.columns:
            df["offset_min"] = df["labresultrevisedoffset"]

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value", "offset_min"])

        df["item_name"] = df["item_name"].astype(str).str.lower()
        df["uom"] = df["labresultuom"] if "labresultuom" in df.columns else ""
        df["event_type"] = "labevent"
        df["has_numeric"] = True

        df = df[["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]]
        print(f"lab: {len(df)} events after basic cleaning.")
        return df

    def load_intake(self, stay_ids):
        path = os.path.join(self.data_dir, "intakeOutput.csv.gz")
        if not os.path.exists(path):
            print(f"WARNING: intakeOutput file not found at {path}")
            return self._empty_df()

        print(f"Loading intakeOutput from {path}...")
        df = pd.read_csv(path, compression="gzip")
        df = df[df["patientunitstayid"].isin(stay_ids)]
        if df.empty:
            print("intakeOutput: no rows for selected stays.")
            return self._empty_df()

        df = df.rename(
            columns={
                "patientunitstayid": "stay_id",
                "intakeoutputoffset": "offset_min",
                "celllabel": "item_name",
                "cellvaluenumeric": "value",
            }
        )

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value", "offset_min"])

        df["item_name"] = df["item_name"].astype(str).str.lower()
        df["uom"] = ""
        df["event_type"] = "intakeevent"
        df["has_numeric"] = True

        df = df[["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]]
        print(f"intakeOutput: {len(df)} events after cleaning.")
        return df

    def load_infusion(self, stay_ids):
        path = os.path.join(self.data_dir, "infusionDrug.csv.gz")
        if not os.path.exists(path):
            print(f"WARNING: infusionDrug file not found at {path}")
            return self._empty_df()

        print(f"Loading infusionDrug from {path}...")
        df = pd.read_csv(path, compression="gzip")
        df = df[df["patientunitstayid"].isin(stay_ids)]
        if df.empty:
            print("infusionDrug: no rows for selected stays.")
            return self._empty_df()

        df = df.rename(
            columns={
                "patientunitstayid": "stay_id",
                "infusionoffset": "offset_min",
                "drugname": "item_name",
                "drugamount": "value",
            }
        )

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value", "offset_min"])

        df["item_name"] = df["item_name"].astype(str).str.lower()
        df["uom"] = ""
        df["event_type"] = "infusionevent"
        df["has_numeric"] = True

        df = df[["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]]
        print(f"infusionDrug: {len(df)} events after cleaning.")
        return df

    def load_med(self, stay_ids):
        path = os.path.join(self.data_dir, "medication.csv.gz")
        if not os.path.exists(path):
            print(f"WARNING: medication file not found at {path}")
            return self._empty_df()

        print(f"Loading medication from {path}...")
        df = pd.read_csv(path, compression="gzip")
        df = df[df["patientunitstayid"].isin(stay_ids)]
        if df.empty:
            print("medication: no rows for selected stays.")
            return self._empty_df()

        df = df.rename(
            columns={
                "patientunitstayid": "stay_id",
                "drugstartoffset": "offset_min",
                "drugname": "item_name",
                "dosage": "raw_dosage",
            }
        )

        # parse dosage into numeric + unit string
        values = []
        uoms = []
        for s in df["raw_dosage"].astype(str):
            v, u = self._parse_dosage(s)
            values.append(v)
            uoms.append(u)
        df["value"] = values
        df["uom"] = uoms

        df = df.dropna(subset=["value", "offset_min"])

        df["item_name"] = df["item_name"].astype(str).str.lower()
        df["event_type"] = "medevent"
        df["has_numeric"] = True

        df = df[["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]]
        print(f"medication: {len(df)} events after dosage parsing & cleaning.")
        return df

    def load_micro(self, stay_ids):
        path = os.path.join(self.data_dir, "microLab.csv.gz")
        if not os.path.exists(path):
            print(f"WARNING: microLab file not found at {path}")
            return self._empty_df()

        print(f"Loading microLab from {path}...")
        df = pd.read_csv(path, compression="gzip")
        df = df[df["patientunitstayid"].isin(stay_ids)]
        if df.empty:
            print("microLab: no rows for selected stays.")
            return self._empty_df()

        df = df.rename(
            columns={
                "patientunitstayid": "stay_id",
                "culturetakenoffset": "offset_min",
            }
        )

        for col in ["culturesite", "organism", "antibiotic", "sensitivitylevel"]:
            if col not in df.columns:
                df[col] = ""

        # item identity is antibiotic (per table_candidates)
        df["item_name"] = df["antibiotic"].astype(str).str.lower()

        df["value"] = float("nan")
        df["uom"] = ""
        df["event_type"] = "microevent"
        df["has_numeric"] = False

        df = df[["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]]
        print(f"microLab: {len(df)} events after cleaning.")
        return df

    def load_treatment(self, stay_ids):
        path = os.path.join(self.data_dir, "treatment.csv.gz")
        if not os.path.exists(path):
            print(f"WARNING: treatment file not found at {path}")
            return self._empty_df()

        print(f"Loading treatment from {path}...")
        df = pd.read_csv(path, compression="gzip")
        df = df[df["patientunitstayid"].isin(stay_ids)]
        if df.empty:
            print("treatment: no rows for selected stays.")
            return self._empty_df()

        df = df.rename(
            columns={
                "patientunitstayid": "stay_id",
                "treatmentoffset": "offset_min",
                "treatmentstring": "item_name",
            }
        )

        df["item_name"] = df["item_name"].astype(str).str.lower()
        df["value"] = float("nan")
        df["uom"] = ""
        df["event_type"] = "treatmentevent"
        df["has_numeric"] = False

        df = df[["stay_id", "offset_min", "event_type", "item_name", "value", "uom", "has_numeric"]]
        print(f"treatment: {len(df)} events after cleaning.")
        return df

    # --------------------------------------------------------
    # Build stay-level data (24h window, LOS ≥ 6h)
    # --------------------------------------------------------

    def _apply_90pct_filter_per_table(self, df, label):
        if df.empty:
            return df
        freq = df["item_name"].value_counts()
        cum = freq.cumsum() / freq.sum()
        keep = set(cum[cum <= 0.90].index)
        before = len(df)
        df = df[df["item_name"].isin(keep)]
        after = len(df)
        print(f"{label}: 90% filter {before} -> {after} events (kept {len(keep)} items).")
        return df

    def process_data(self):
        icu_df = self.load_patient()
        if icu_df.empty:
            print("No stays in patient table.")
            return []

        stays = set(icu_df["stay_id"])

        lab = self.load_labs(stays)
        intake = self.load_intake(stays)
        infusion = self.load_infusion(stays)
        med = self.load_med(stays)
        micro = self.load_micro(stays)
        treat = self.load_treatment(stays)

        # per-table 90% frequency filtering
        lab = self._apply_90pct_filter_per_table(lab, "lab")
        intake = self._apply_90pct_filter_per_table(intake, "intakeOutput")
        infusion = self._apply_90pct_filter_per_table(infusion, "infusionDrug")
        med = self._apply_90pct_filter_per_table(med, "medication")
        micro = self._apply_90pct_filter_per_table(micro, "microLab")
        treat = self._apply_90pct_filter_per_table(treat, "treatment")

        all_df = pd.concat([lab, intake, infusion, med, micro, treat], ignore_index=True)
        if all_df.empty:
            print("No events left after filtering.")
            return []

        # absolute time
        intime_map = icu_df.set_index("stay_id")["intime"].to_dict()
        all_df["time"] = pd.to_timedelta(all_df["offset_min"], unit="m") + all_df["stay_id"].map(intime_map)
        all_df = all_df.sort_values(["stay_id", "time"])

        # LOS ≥ 6h
        last_event = all_df.groupby("stay_id")["time"].max()
        icu_df = icu_df.merge(last_event.rename("last_time"), on="stay_id", how="left")
        icu_df = icu_df.dropna(subset=["last_time"])

        icu_df["los_hours"] = (icu_df["last_time"] - icu_df["intime"]).dt.total_seconds() / 3600.0
        icu_df = icu_df[icu_df["los_hours"] >= 6]

        print(f"Stays after LOS >= 6h filter: {len(icu_df)}")

        grouped = all_df.groupby("stay_id")
        data = []

        for _, row in tqdm(icu_df.iterrows(), total=len(icu_df), desc="Collecting stay event streams"):
            sid = row["stay_id"]
            intime = row["intime"]
            cutoff = intime + pd.Timedelta(hours=72)

            if sid not in grouped.groups:
                continue

            ev = grouped.get_group(sid)
            ev = ev[(ev["time"] >= intime) & (ev["time"] <= cutoff)]
            if ev.empty:
                continue

            data.append(
                dict(
                    stay_id=sid,
                    demographics=row,
                    events=ev,
                    intime=intime,
                )
            )

        print(f"Total stays with usable 72h data: {len(data)}")
        return data

    # --------------------------------------------------------
    # Sequence creation
    # --------------------------------------------------------

    def create_sequences(self, data, split):
        sequences = []
        eval_items = []
        shard_idx = 0

        for entry in tqdm(data, desc=f"Creating sequences ({split})"):
            sid = entry["stay_id"]
            demo = entry["demographics"]
            events = entry["events"]

            gender = str(demo["gender"]).lower()
            age = str(demo["anchor_age"])
            status = str(demo.get("discharge_status", "")).lower()
            demo_text = f"gender {gender} age {age} status {status}"
            demo_tok = self.tokenizer.encode(demo_text, add_special_tokens=False)

            curr_ids = demo_tok.copy()
            curr_types = [0] * len(curr_ids)

            for _, e in events.iterrows():
                etype = e["event_type"]
                has_num = bool(e["has_numeric"])
                is_lab = etype == "labevent"
                supervise = has_num and is_lab  # labs only

                t_tok = self.tokenizer.encode(self.get_time_tokens(e["offset_min"]), add_special_tokens=False)
                etype_id = self.tokenizer.convert_tokens_to_ids(etype)
                item_tok = self.tokenizer.encode(self._item_name(etype, e["item_name"]), add_special_tokens=False)

                if has_num:
                    v_str = self.format_value(e["value"])
                    if v_str is None:
                        continue
                    v_tok = self.tokenizer.encode(v_str, add_special_tokens=False)
                    uom_str = self._item_unit(etype, e.get("uom", ""))
                    uom_tok = self.tokenizer.encode(
                        (uom_str + " [EOE]") if uom_str else "[EOE]",
                        add_special_tokens=False,
                    )

                    event_toks = t_tok + [etype_id] + item_tok + v_tok + uom_tok
                    type_ids = (
                        [0] * len(t_tok)
                        + [0]
                        + [0] * len(item_tok)
                        + ([1] * len(v_tok) if supervise else [0] * len(v_tok))
                        + [0] * len(uom_tok)
                    )
                else:
                    uom_tok = self.tokenizer.encode("[EOE]", add_special_tokens=False)
                    event_toks = t_tok + [etype_id] + item_tok + uom_tok
                    type_ids = [0] * len(event_toks)

                # overflow handling
                if len(curr_ids) + len(event_toks) > self.max_len:
                    sequences.append(dict(stay_id=sid, input_ids=curr_ids, type_ids=curr_types))
                    curr_ids = demo_tok.copy()
                    curr_types = [0] * len(curr_ids)

                # eval items (labs only) for val/test
                if supervise and split in ["val", "test"]:
                    prompt = curr_ids + t_tok + [etype_id] + item_tok
                    if len(prompt) > self.max_len:
                        prompt = demo_tok + prompt[-(self.max_len - len(demo_tok)):]
                    eval_items.append(
                        dict(
                            stay_id=sid,
                            prompt_ids=prompt,
                            label_ids=v_tok + uom_tok,
                            valuenum=e["value"],
                            item_name=e["item_name"],
                            event_type=etype,
                        )
                    )

                curr_ids.extend(event_toks)
                curr_types.extend(type_ids)

            if len(curr_ids) > len(demo_tok):
                sequences.append(dict(stay_id=sid, input_ids=curr_ids, type_ids=curr_types))

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
            print(f"Saved {split}_eval.pkl: {len(eval_items)} items")

    # --------------------------------------------------------
    # Shard helpers
    # --------------------------------------------------------

    def _write_shard(self, data, split, idx):
        path = os.path.join(self.out_dir, f"{split}_shard_{idx}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _merge_shards(self, split, n):
        all_data = []
        for i in range(n):
            path = os.path.join(self.out_dir, f"{split}_shard_{i}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    all_data.extend(pickle.load(f))
                os.remove(path)

        out_path = os.path.join(self.out_dir, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"Saved {split}.pkl: {len(all_data)} sequences")

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self):
        data = self.process_data()
        if not data:
            print("No usable stays.")
            return

        stays = [d["stay_id"] for d in data]

        if len(stays) < 10:
            print("Too few stays, using all for train.")
            train, val, test = data, [], []
        else:
            train_ids, tmp_ids = train_test_split(stays, test_size=0.3, random_state=42)
            val_ids, test_ids = train_test_split(tmp_ids, test_size=0.5, random_state=42)
            train = [d for d in data if d["stay_id"] in train_ids]
            val = [d for d in data if d["stay_id"] in val_ids]
            test = [d for d in data if d["stay_id"] in test_ids]

        print(f"Train stays={len(train)}, val={len(val)}, test={len(test)}")

        self.create_sequences(train, "train")
        if val:
            self.create_sequences(val, "val")
        if test:
            self.create_sequences(test, "test")

        print("eICU LabTOP preprocessing complete.")


if __name__ == "__main__":
    data_dir = "/Users/ziwei/physionet.org/files/eicu-crd/2.0"
    out_dir = "eicu_labtop_processed"
    os.makedirs(out_dir, exist_ok=True)

    pre = LabTOPEICUPreprocessor(
        data_dir=data_dir,
        out_dir=out_dir,
        max_len=MAX_LEN,
        stay_limit=STAY_LIMIT,
        shard_size=SHARD_SIZE,
    )
    pre.run()
