#!/usr/bin/env python3
"""Read-only proof of the exact historical NVIDIA wheel metadata defect."""
import base64,csv,hashlib,importlib.metadata as md,json,platform,struct
from pathlib import Path
d=md.distribution("nvidia-cusparselt-cu13")
wheel=Path(d.locate_file("nvidia_cusparselt_cu13-0.8.1.dist-info/WHEEL"))
record=Path(d.locate_file("nvidia_cusparselt_cu13-0.8.1.dist-info/RECORD"))
name="nvidia/cusparselt/lib/libcusparseLt.so.0";library=Path(d.locate_file(name))
rows=list(csv.reader(record.read_text().splitlines()))
row=next(r for r in rows if r[0]==name)
before=library.stat();h=hashlib.sha256()
with library.open("rb") as f:
 header=f.read(64);h.update(header)
 while part:=f.read(4*1024*1024):h.update(part)
after=library.stat()
out={"package":d.metadata["Name"],"version":d.version,"system":platform.system(),"machine":platform.machine(),
"wheel_path":str(wheel),"wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),
"tags":[s.removeprefix("Tag: ") for s in wheel.read_text().splitlines() if s.startswith("Tag: ")],
"library_path":str(library),"library_bytes":before.st_size,"library_sha256":h.hexdigest(),
"record_library":row,"elf_magic":header[:4].hex(),"elf_class":header[4]*32,
"elf_machine":struct.unpack("<H",header[18:20])[0],
"stable_library":(before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
"record_matches_library":row[1]=="sha256="+base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()}
print(json.dumps(out,sort_keys=True))
