"""Download the official reduced dataset and inspect metadata without reading flows."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


def fetch_chunk(folder, config, start, stop):
    path = folder / f"{start:012d}-{stop:012d}.bin"
    receipt_path = path.with_suffix(".json")
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if path.stat().st_size == stop - start + 1 and hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]:
            return receipt
        raise RuntimeError("Retained download chunk failed its checksum")
    request = urllib.request.Request(config["dataset_url"], headers={"Range": f"bytes={start}-{stop}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes {start}-{stop}/{config['dataset_bytes']}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError("Official range response does not match the requested chunk")
        data = response.read(stop - start + 2)
        etag = response.headers.get("ETag")
    if len(data) != stop - start + 1:
        raise RuntimeError("Short range response. Completed chunks remain reusable.")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    receipt = {"start": start, "stop": stop, "etag": etag,
               "sha256": hashlib.sha256(data).hexdigest()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    import tables
    root = args.root
    config = json.loads((root / "configs/pilot-v1.json").read_text())
    folder = root / "data"
    folder.mkdir(exist_ok=True)
    target = folder / "CESNET-QUICEXT-25-XS.h5"
    partial = target.with_suffix(".partial")
    size = config["dataset_bytes"]
    if not target.exists():
        if shutil.disk_usage(folder).free < size * 2 + 2_000_000_000:
            raise RuntimeError("Dataset size or available disk does not meet the bounded download plan")
        chunks = folder / "download-chunks"
        chunks.mkdir(exist_ok=True)
        chunk_size = 2 * 1024 * 1024
        bounds = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
        receipts = []
        failures = []
        # Small bounded ranges avoid the premature EOF seen in the first run.
        # The sixteen workers perform I/O only. Each completed range is verified
        # and can be reused after a scheduler interruption.
        with ThreadPoolExecutor(max_workers=16) as executor:
            tasks = {executor.submit(fetch_chunk, chunks, config, start, stop): (start,stop) for start,stop in bounds}
            last_report = time.monotonic()
            for task in as_completed(tasks):
                try:
                    receipts.append(task.result())
                except Exception as error:
                    start, stop = tasks[task]
                    failures.append({"start":start,"stop":stop,"error_type":type(error).__name__,"reason":str(error)[:180]})
                if time.monotonic() - last_report > 20:
                    print(json.dumps({"completed_chunks": len(receipts), "failed_chunks": len(failures), "total_chunks": len(bounds)}), flush=True)
                    last_report = time.monotonic()
        if failures:
            out = root / "outputs"
            out.mkdir(exist_ok=True)
            (out / f"download-failures-{time.time_ns()}.json").write_text(json.dumps(failures,indent=2)+"\n")
            raise RuntimeError(f"{len(failures)} chunks failed. Completed chunks retained for an explicit retry.")
        if len({r["etag"] for r in receipts}) != 1 or not receipts[0]["etag"]:
            raise RuntimeError("Source ETag changed between range requests")
        with partial.open("wb") as output:
            for start, stop in bounds:
                with (chunks / f"{start:012d}-{stop:012d}.bin").open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        if partial.stat().st_size != size:
            raise RuntimeError("Incomplete download retained for explicit resumption")
        partial.replace(target)
    if target.stat().st_size != size:
        raise RuntimeError("Existing dataset size differs from the verified source")
    digest = hashlib.sha256()
    with target.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    with tables.open_file(target, mode="r") as database:
        nodes = sorted(database.root.flows, key=lambda n: n._v_name)
        inventory = {
            "status": "METADATA_ONLY_NO_FLOW_ROWS_READ",
            "dataset_url": config["dataset_url"], "bytes": size, "sha256": digest.hexdigest(),
            "tables": [{"path": n._v_pathname, "rows": int(n.nrows)} for n in nodes],
            "columns": {name: str(dtype) for name, dtype in nodes[0].coldtypes.items()},
            "apps": {name: int(value) for name, value in nodes[0].get_enum("APP")._names.items()},
        }
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"status": inventory["status"], "sha256": digest.hexdigest(), "date_tables": len(nodes)}), flush=True)


if __name__ == "__main__":
    main()
