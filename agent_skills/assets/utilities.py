"""
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are
 proprietary to SiMa and may be covered by U.S. and Foreign Patents,
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is
 strictly forbidden unless prior written permission is obtained from
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who
 have executed Confidentiality and Non-disclosure agreements explicitly
 covering such access.

 The copyright notice above does not evidence any actual or intended
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.

**************************************************************************
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import argparse
import io
import logging
import shutil
import sys
import tarfile
import tempfile
import onnx


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}

def _replace_pipeline_sequence_lines_in_archive(
    archive_path: Path, start_line: int, end_line: int, replacement_line: str
) -> None:
    """
    Replace a line range in pipeline_sequence.json inside a .tar.gz archive.
    """
    if start_line < 1 or end_line < start_line:
        raise ValueError(
            f"Invalid line range start={start_line}, end={end_line}"
        )

    archive_path = archive_path.resolve()
    with tempfile.TemporaryDirectory(prefix="pipeline_seq_edit_") as temp_dir:
        patched_archive = Path(temp_dir) / archive_path.name
        pipeline_file_found = False

        with tarfile.open(str(archive_path), "r:gz") as src_tar:
            with tarfile.open(str(patched_archive), "w:gz") as dst_tar:
                for member in src_tar.getmembers():
                    if not member.isfile():
                        dst_tar.addfile(member)
                        continue

                    source_file = src_tar.extractfile(member)
                    if source_file is None:
                        raise RuntimeError(f"Failed to read archive member: {member.name}")

                    if Path(member.name).name != "pipeline_sequence.json":
                        try:
                            dst_tar.addfile(member, source_file)
                        finally:
                            source_file.close()
                        continue

                    with source_file:
                        content = source_file.read().decode("utf-8")
                    lines = content.splitlines(keepends=True)

                    if len(lines) < end_line:
                        raise ValueError(
                            "pipeline_sequence.json has fewer lines than requested range: "
                            f"{len(lines)} < {end_line}"
                        )

                    updated_lines = (
                        lines[: start_line - 1]
                        + [replacement_line + "\n"]
                        + lines[end_line:]
                    )
                    updated_bytes = "".join(updated_lines).encode("utf-8")

                    patched_member = tarfile.TarInfo(name=member.name)
                    patched_member.size = len(updated_bytes)
                    patched_member.mode = member.mode
                    patched_member.mtime = member.mtime
                    patched_member.uid = member.uid
                    patched_member.gid = member.gid
                    patched_member.uname = member.uname
                    patched_member.gname = member.gname
                    patched_member.type = tarfile.REGTYPE
                    dst_tar.addfile(patched_member, io.BytesIO(updated_bytes))
                    pipeline_file_found = True

        if not pipeline_file_found:
            raise FileNotFoundError(
                f"pipeline_sequence.json was not found in archive: {archive_path}"
            )

        shutil.move(str(patched_archive), str(archive_path))



def _list_image_files(folder_path: Path) -> List[Path]:
    """
    Return a list of image file paths in the specified folder.
    """
    folder = folder_path
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    image_paths: List[Path] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(entry)

    return image_paths




def _prepare_results_dir(build_dir: Path, model_path: Path) -> Tuple[Path, str]:
    """
    Create a clean results directory under build_dir named after model_path stem.
    """
    output_model_name = model_path.stem
    build_dir_path = build_dir.resolve()
    results_dir = (build_dir_path / output_model_name).resolve()

    if results_dir.exists():
        if results_dir.is_dir():
            print(f"Removing existing directory: {results_dir}", flush=True)
            shutil.rmtree(results_dir)
        else:
            raise NotADirectoryError(
                f"Path exists but is not a directory: {results_dir}"
            )

    results_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results will be written to {results_dir}", flush=True)
    return results_dir, output_model_name



