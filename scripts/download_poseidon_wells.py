"""Download the Poseidon 3D (Browse Basin, NW Australia) well data used for AVO calibration.

All data is open-file data released by Geoscience Australia (CC BY 3.0 AU); the authoritative source is
NOPIMS (https://public.neats.nopta.gov.au/nopims). This script pulls it from mirrors that allow scripted
downloads:

  GitHub  WesleyTheGeolien/t20-Dash_Well_Viz (Apache-2.0): Occam spliced composite LAS for the 8 wells,
          well headers, Poseidon-1 and Kronos-1 formation tops (from the well completion reports).
  GitHub  aldimhd/Shearwave_Prediction: VSP-calibrated depth logs (Kronos-1, Pharos-1, Proteus-1 ST2)
          and Poseidon-1/-2 log CSVs.
  GitHub  Firzalds/Gabor-Deconvolution-and-DTW-for-Automatic-Well-Seismic-Tie: Torosa-3 logs,
          deviation, VSP checkshots and markers.
  GA      Boreholes WFS (https://services.ga.gov.au/gis/boreholes/wfs): borehole headers and
          lithostratigraphic intervals (formation tops).
  Drive   (--drive) ConocoPhillips' Google Drive share of the full well packages: checkshot reports,
          deviation surveys and well completion reports. Often blocked by Google's download quota.

Output in data/poseidon/wells/:
  logs/<well>/            spliced composite LAS (main log set, one per well)
  logs_extra/<well>/      VSP-calibrated logs, CSV logs, Torosa-3 package
  ga/                     GA borehole headers and stratigraphy intervals
  tops/well_tops.csv      combined formation tops (MD below rig datum)
  tops/t20/               tops and well headers from the t20 repository
  checkshots/, deviation/, reports/   (--drive only)

Files already present are skipped, so the script can be rerun to fill gaps.

    python scripts/download_poseidon_wells.py [--drive] [--out data/poseidon/wells]
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "poseidon" / "wells"

T20 = "https://raw.githubusercontent.com/WesleyTheGeolien/t20-Dash_Well_Viz/master"
SWP = "https://raw.githubusercontent.com/aldimhd/Shearwave_Prediction/master/data"
TOROSA3 = (
    "https://raw.githubusercontent.com/Firzalds/Gabor-Deconvolution-and-DTW-for-Automatic-Well-Seismic-Tie"
    "/main/Data/Well%20Data%20Torosa-03"
)
GA_WFS = "https://services.ga.gov.au/gis/boreholes/wfs"

# (relative destination, url)
MIRROR_FILES = [
    ("logs/Boreas-1/Boreas1Decim.LAS", f"{T20}/Data/Boreas1Decim.LAS"),
    ("logs/Kronos-1/Kronos1Decim.LAS", f"{T20}/Data/Kronos1Decim.LAS"),
    ("logs/Pharos-1/Pharos1Decim.LAS", f"{T20}/Data/Pharos1Decim.LAS"),
    ("logs/Poseidon-1/Poseidon1Decim.LAS", f"{T20}/Data/Poseidon1Decim.LAS"),
    ("logs/Poseidon-2/Poseidon2Decim.LAS", f"{T20}/Data/Poseidon2Decim.LAS"),
    ("logs/Poseidon-North-1/PoseidonNorth1Decim.LAS", f"{T20}/Data/PoseidonNorth1Decim.LAS"),
    ("logs/Proteus-1/Proteus1Decim.LAS", f"{T20}/Data/Proteus1Decim.LAS"),
    ("logs/Torosa-1/Torosa1LDecim.LAS", f"{T20}/Data/Torosa1LDecim.LAS"),
    ("tops/t20/Poseidon1_tops.csv", f"{T20}/Data/Poseidon1_tops.csv"),
    ("tops/t20/kronos_1_strat.csv", f"{T20}/Data/kronos_1_strat.csv"),
    ("tops/t20/Well_Headers.csv", f"{T20}/Data/Well_Headers.csv"),
    ("logs_extra/Kronos-1/CP_Kronos1_Logs_depth.las", f"{SWP}/CP_Kronos1_Logs_depth.las"),
    ("logs_extra/Pharos-1/COP_Pharos-1_logs_depth.las", f"{SWP}/COP_Pharos-1_logs_depth.las"),
    ("logs_extra/Proteus-1/COP_Proteus-1ST2_logs_depth_MD.las", f"{SWP}/COP_Proteus-1ST2_logs_depth_MD.las"),
    ("logs_extra/Poseidon-1/poseidon1.csv", f"{SWP}/poseidon1.csv"),
    ("logs_extra/Poseidon-2/poseidon2.csv", f"{SWP}/poseidon2.csv"),
    ("logs_extra/Torosa-3/Torosa-3_logs.las", f"{TOROSA3}/Torosa-3_logs.las"),
    ("logs_extra/Torosa-3/Torosa-3_logs_Time.las", f"{TOROSA3}/Torosa-3_logs_Time.las"),
    ("logs_extra/Torosa-3/Dev_Torosa3.txt", f"{TOROSA3}/Dev_Torosa3.txt"),
    ("logs_extra/Torosa-3/VSP_Torosa3.txt", f"{TOROSA3}/VSP_Torosa3.txt"),
    ("logs_extra/Torosa-3/Marker.xlsx", f"{TOROSA3}/Marker.xlsx"),
]

GA_NAME_FILTER = " OR ".join(
    f"{{field}} ILIKE '{name}%'" for name in ("Poseidon", "Kronos", "Pharos", "Proteus", "Boreas", "Torosa")
)

# (category, well, file name, Google Drive file id) -- ConocoPhillips share, used with --drive
DRIVE_FILES = [
    ("checkshots", "Kronos-1", "CP_Kronos1_Logs_time.las", "11mk83xwyzeExCfmiQBrLW6V5GfcbrWoC"),
    ("checkshots", "Kronos-1", "CP_Kronos1_Time_to_Depth_Report.xls", "161awfXRE6msjDN5KJhtITdGN5_SAJvpi"),
    ("checkshots", "Kronos-1", "CP_Kronos1_Velocity_Report.xls", "1h2D5xPhDtvcrX6XPziS8tm4zWR-sK0t4"),
    ("checkshots", "Pharos-1", "COP-Pharos-1_QC_VSP_TimeDepthListing.xlsx", "19-OVRAvSabQoBnikvSeX4hXQc58ZbXuN"),
    ("checkshots", "Pharos-1", "COP_Pharos-1_logs_time.las", "1UF2ZJgosk3SuC8ZtNISBAsMEFQdWTlhh"),
    ("checkshots", "Poseidon-2", "CP_Poseidon2_Time_to_Depth_Report.xls", "1cNFpYYoRWbjSw_zlCwhyabZXyCAX7169"),
    ("checkshots", "Poseidon-2", "CP_Poseidon2_Velocity_Report.xls", "1-tuMT1Zg9Ju4UWgsrAvOPSG4sVc7rsf4"),
    ("checkshots", "Poseidon-2", "CP_Poseidon2_logs_time.las", "1ykXfo_neZk5oOj1faIP_y1-1EeLik4lD"),
    ("checkshots", "Poseidon-2", "ConocoPhillips_Poseidon-2_Suite2_Run8_VSP_VSI4-GR_Data.xls",
     "1KoTkAx3wwOZXkz8sPMQy_D5qnPKRjlhj"),
    ("checkshots", "Torosa-1", "TZV_DEPTH_CHECKSHOT.XLS", "1Lkf41wP_jvfsW7oVKs2GOhXCH0Q9WaxI"),
    ("checkshots", "Torosa-1", "VSPZO_COMPUTED_VSPCS.LAS", "1xwJkxybA14EtFjAsMtBk64oKppm7K0IO"),
    ("deviation", "Pharos-1", "Pharos-1_Definitive Survey Report_TD 5220m.xlsx", "1LOs46I0HtpMFqRKa4epC1Yi_iHcENBBU"),
    ("deviation", "Poseidon-North-1", "COP_Nth1_Definitive Survey Report_TD 5287m.xlsx",
     "13-85gHzOWd0v7_vQD2JEMhC4nnPd9uAD"),
    ("deviation", "Proteus-1", "Proteus Surveys.xlsx", "1uk-CHxyTSRpG8pn-WS76bb1iCcW3BVnu"),
    ("deviation", "Proteus-1", "Proteus-1 ST2_Definitive Survey Report_TD 5250m.xlsx",
     "1Q8UuNlaYMNyjoNRldE7lVio93EFiR0gO"),
    ("deviation", "Torosa-1", "Torosa-1_Surveys_4685mMD.txt", "1yv2FiU-khbiqxt_tHKdTbDgXi-KRnrPi"),
    ("reports", "Kronos-1", "Kronos-1 WCR_Interp_Rev0.pdf", "1ILnEOaMMXvPRmBGehCmclypTyztMT9Sw"),
    ("reports", "Kronos-1", "Kronos1_Basic WCR_Final.pdf", "1f429IhGnS5bH-Ueocc5tTcQBWJUn0ZUL"),
    ("reports", "Pharos-1", "Pharos-1 End Of Well Report_Final.pdf", "1XtfunRohXp4HYPoyT-V6N5msK2IUYXNo"),
    ("reports", "Pharos-1", "Pharos-1_Basic WCR_Rev1.pdf", "1eZujmAUHAZYtb39HM6bdtSg5Y182ZbI5"),
    ("reports", "Poseidon-1", "Poseidon_1_WCR_Volume_2_Interpretive_Data.pdf", "1xUoHTACdBJSC_ml8aM42vvBZ2MnEa5aT"),
    ("reports", "Poseidon-2", "Poseidon-2 WCR_Basic_final.pdf", "11HPn4qxUHK3juWV8NKO_9TOSPCri8cSo"),
    ("reports", "Poseidon-2", "Poseidon-2 WCR_Interp_Final.pdf", "1lx9rXUTReRGDzd4QZtRKQC3wwZKf5ESc"),
    ("reports", "Poseidon-North-1", "Pos North1_Basic WCR rev0.pdf", "1swoDAxxgIlNbkIx6gAw0XSg3GZuZS7hh"),
    ("reports", "Proteus-1", "Proteus-1_Basic WCR_Final.pdf", "15J12yMGEkrSu09eXkFBez50eQ35KfCaf"),
    ("reports", "Torosa-1", "Torosa-1 WCR_Basic Data.pdf", "1O8F5uInxGyo1PWOR0rNnSIprwBozmMQ3"),
    ("reports", "Torosa-1", "Torosa-1 WCR_Interpretive Data.pdf", "1oC5giB47WsqO5hCazVudeUGYVsCepZoa"),
    ("tops", "Poseidon-2", "Lithofacies Top Depths_Poseidon2.xls", "1EfbcglkUXBqsFx6Hhd2NnA82qH1PbC73"),
]

# GA biozone / sequence codes (e.g. T20, K40, Td_Tr10) are not lithostratigraphic tops.
_SEQUENCE_CODE = re.compile(r"^(T|K|J|Td|Tr)\d|^Td_|^T\d", re.IGNORECASE)


class QuotaExceeded(RuntimeError):
    pass


def _fetch(opener, url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "force-hackathon-avo/0.1"})
    with opener.open(request, timeout=timeout) as response:
        return response.read()


def _save(opener, url: str, dest: Path, label: str) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        data = _fetch(opener, url)
    except Exception as exc:
        print(f"FAILED   {label}: {exc}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"OK {len(data) / 1e6:7.2f} MB  {label}")
    return True


def _drive_download(opener, file_id: str) -> bytes:
    query = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    data = _fetch(opener, f"https://drive.usercontent.google.com/download?{query}")
    head = data[:512].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        if b"quota exceeded" in data.lower():
            raise QuotaExceeded("Google Drive download quota exceeded")
        raise RuntimeError("Google Drive returned an HTML page instead of the file")
    return data


def _ga_csv(opener, type_name: str, name_field: str) -> bytes:
    query = urllib.parse.urlencode({
        "SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature", "TYPENAME": type_name,
        "outputFormat": "csv", "CQL_FILTER": GA_NAME_FILTER.format(field=name_field),
    })
    return _fetch(opener, f"{GA_WFS}?{query}")


def _well_id(name: str) -> str:
    name = re.sub(r"\s+ST\d+$", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[\s_]+", "-", name).replace("Kronons", "Kronos")


def build_tops(out: Path) -> Path:
    """Combine t20 (well completion report) tops with GA lithostratigraphy into one table."""
    rows: list[dict] = []
    t20_wells = set()
    for path, well in ((out / "tops/t20/Poseidon1_tops.csv", "Poseidon-1"),
                       (out / "tops/t20/kronos_1_strat.csv", "Kronos-1")):
        if not path.is_file():
            continue
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        reader = csv.reader(lines)
        header = [h.strip().lower() for h in next(reader)]
        name_col = header.index("formation_name") if "formation_name" in header else header.index("formation name")
        md_col = header.index("mdrt")
        for record in reader:
            try:
                md = float(record[md_col])
            except (ValueError, IndexError):
                continue
            rows.append({"WELL": well, "SURFACE": record[name_col].strip(), "MD_M": md,
                         "SOURCE": "t20 repo (well completion report)"})
        t20_wells.add(well)

    strat = out / "ga" / "stratigraphy_logs.csv"
    if strat.is_file():
        with strat.open(encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        by_well: dict[str, list[dict]] = {}
        for record in records:
            if "TOP HOLE" in record["BOREHOLE_NAME"].upper():
                continue
            by_well.setdefault(_well_id(record["BOREHOLE_NAME"]), []).append(record)
        for well, records in sorted(by_well.items()):
            if well in t20_wells:
                continue
            wcr = [r for r in records if "WCR" in r["INTERVAL_COLLECTION_NAME"]]
            chosen = wcr or records
            seen = set()
            for record in sorted(chosen, key=lambda r: float(r["INTERVAL_BEGIN_M"])):
                unit = (record["STRAT_UNIT_NAME"] or "").strip()
                if not unit or _SEQUENCE_CODE.match(unit):
                    continue
                md = round(float(record["INTERVAL_BEGIN_M"]), 2)
                if (unit, md) in seen:
                    continue
                seen.add((unit, md))
                rows.append({"WELL": well, "SURFACE": unit, "MD_M": md,
                             "SOURCE": f"GA Boreholes WFS ({record['INTERVAL_COLLECTION_NAME'].strip()})"})

    dest = out / "tops" / "well_tops.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["WELL"], r["MD_M"]))
    with dest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ["WELL", "SURFACE", "MD_M", "SOURCE"])
        writer.writeheader()
        writer.writerows(rows)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    parser.add_argument("--drive", action="store_true",
                        help="also fetch checkshot reports, deviation surveys and well reports from Google Drive")
    args = parser.parse_args(argv)
    out = args.out
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    failed = 0

    for rel, url in MIRROR_FILES:
        failed += not _save(opener, url, out / rel, rel)

    for type_name, field, name in (("bh:Boreholes", "NAME", "boreholes.csv"),
                                   ("bh:BoreholeStratigraphyLogs", "BOREHOLE_NAME", "stratigraphy_logs.csv")):
        dest = out / "ga" / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        try:
            data = _ga_csv(opener, type_name, field)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"OK {len(data) / 1e6:7.2f} MB  ga/{name}")
        except Exception as exc:
            failed += 1
            print(f"FAILED   ga/{name}: {exc}")

    tops = build_tops(out)
    with tops.open(encoding="utf-8") as stream:
        wells = sorted({row["WELL"] for row in csv.DictReader(stream)})
    print(f"Tops table: {tops.relative_to(out)} ({', '.join(wells)})")

    if args.drive:
        quota = False
        for category, well, name, file_id in DRIVE_FILES:
            dest = out / category / well / name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            try:
                data = _drive_download(opener, file_id)
            except QuotaExceeded as exc:
                quota, failed = True, failed + 1
                print(f"FAILED   {category}/{well}/{name}: {exc}")
                continue
            except Exception as exc:
                failed += 1
                print(f"FAILED   {category}/{well}/{name}: {exc}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"OK {len(data) / 1e6:7.2f} MB  {category}/{well}/{name}")
            time.sleep(1)
        if quota:
            print("Google Drive quota is exceeded for some shared files; rerun with --drive later.")

    print(f"\nDone -> {out} ({failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
