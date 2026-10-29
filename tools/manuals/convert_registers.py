#!/usr/bin/env python3
"""
Convert AMD AIE-ML register reference HTML files to compact, LLM-readable text format.

Parses the HTML documentation and outputs a hierarchical folder structure with
compact register definitions optimized for token efficiency.

Usage:
    python convert_registers.py am025-versal-aie-ml-register-reference/html output_dir
"""

import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from collections import defaultdict


class RegisterHTMLParser(HTMLParser):
    """Parse a single register HTML file and extract structured data."""

    def __init__(self):
        super().__init__()
        self.in_td = False
        self.in_th = False
        self.in_span = False
        self.current_data = []
        self.rows = []
        self.current_row = []
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.in_td = True
            self.current_data = []
        elif tag == "th":
            self.in_th = True
            self.current_data = []
        elif tag == "title":
            self.in_title = True
            self.current_data = []
        elif tag == "tr":
            self.current_row = []
        elif tag == "span":
            attrs_dict = dict(attrs)
            if "class" in attrs_dict and "tooltip" in attrs_dict["class"]:
                self.in_span = True

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_td = False
            self.current_row.append("".join(self.current_data).strip())
        elif tag == "th":
            self.in_th = False
            self.current_row.append("".join(self.current_data).strip())
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
        elif tag == "title":
            self.in_title = False
            self.title = "".join(self.current_data).strip()
        elif tag == "span":
            self.in_span = False

    def handle_data(self, data):
        if self.in_span:
            return
        if self.in_td or self.in_th or self.in_title:
            self.current_data.append(data)


def parse_register_html(filepath: Path) -> dict | None:
    """Parse an HTML file and return register data."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None

    parser = RegisterHTMLParser()
    try:
        parser.feed(content)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return None

    register = {
        "name": "", "offset": "", "width": "", "type": "",
        "reset": "", "description": "", "fields": []
    }

    for row in parser.rows:
        if len(row) >= 2:
            key = row[0].lower()
            value = row[1]
            if "register name" in key:
                register["name"] = value
            elif "offset" in key:
                register["offset"] = value
            elif "width" in key:
                register["width"] = value
            elif "type" in key and "reset" not in key:
                register["type"] = value
            elif "reset" in key:
                register["reset"] = value
            elif "description" in key:
                register["description"] = value

    for row in parser.rows:
        if len(row) == 5 and row[0].lower() != "field name":
            register["fields"].append({
                "name": row[0], "bits": row[1], "type": row[2],
                "reset": row[3], "description": row[4]
            })

    if not register["name"] and not register["offset"]:
        return None
    return register


def format_register_compact(reg: dict) -> str:
    """Format a register in compact, LLM-readable format."""
    lines = []
    reset_str = f" reset={reg['reset']}" if reg['reset'] and reg['reset'] != "0x00000000" else ""
    desc_str = f' "{reg["description"]}"' if reg["description"] else ""
    lines.append(f"{reg['name']} @ {reg['offset']} [{reg['width']} {reg['type']}{reset_str}]{desc_str}")

    for field in reg["fields"]:
        bits = field["bits"].ljust(7)
        name = field["name"].ljust(28)
        lines.append(f"  {bits} {name} {field['type']}  \"{field['description']}\"")

    return "\n".join(lines)


def tokenize_name(name: str) -> list[str]:
    """
    Split register name into semantic tokens.
    Strips trailing numbers and consolidates multi-word terms.
    """
    parts = name.lower().split("_")
    tokens = []

    i = 0
    while i < len(parts):
        part = parts[i]

        # Consolidate known multi-word terms
        if part == "stream" and i + 1 < len(parts) and parts[i + 1] == "switch":
            tokens.append("stream_switch")
            i += 2
            continue
        elif part == "combo" and i + 1 < len(parts) and parts[i + 1] == "event":
            tokens.append("combo_event")
            i += 2
            continue
        elif part == "edge" and i + 1 < len(parts) and parts[i + 1] == "detection":
            tokens.append("edge_detection")
            i += 2
            continue
        elif part == "tile" and i + 1 < len(parts) and parts[i + 1] == "control":
            tokens.append("tile_control")
            i += 2
            continue
        elif part == "module" and i + 1 < len(parts) and parts[i + 1] == "clock":
            tokens.append("module_clock")
            i += 2
            continue
        elif part == "memory" and i + 1 < len(parts) and parts[i + 1] == "control":
            tokens.append("memory_control")
            i += 2
            continue
        elif part == "error" and i + 1 < len(parts) and parts[i + 1] == "halt":
            tokens.append("error_halt")
            i += 2
            continue
        elif part == "deterministic" and i + 1 < len(parts) and parts[i + 1] == "merge":
            tokens.append("deterministic_merge")
            i += 2
            continue
        elif part == "adaptive" and i + 2 < len(parts) and parts[i + 1] == "clock" and parts[i + 2] == "gate":
            tokens.append("adaptive_clock_gate")
            i += 3
            continue

        # Strip numbers from tokens like "bd0", "r0", "amhh0"
        if re.match(r"^[a-z]+\d+$", part):
            prefix = re.match(r"^([a-z]+)", part).group(1)
            tokens.append(prefix)
            i += 1
            continue

        # Skip pure numbers, part1/part2, slots
        if re.match(r"^\d+$", part) or part in ("part1", "part2") or re.match(r"^slot\d+$", part):
            i += 1
            continue

        tokens.append(part)
        i += 1

    return tokens


def compute_grouping(registers: list[dict], min_per_file: int = 3, max_per_file: int = 100) -> dict:
    """
    Algorithmically compute grouping for registers.

    Returns: dict mapping (folder_path, filename) -> list of registers
    """
    # First, tokenize all register names
    for reg in registers:
        reg["_tokens"] = tokenize_name(reg["_filename"])

    # Try progressively deeper groupings until we get reasonable file sizes
    def group_by_depth(regs, depth):
        """Group registers by first `depth` tokens."""
        groups = defaultdict(list)
        for reg in regs:
            tokens = reg["_tokens"]
            key = tuple(tokens[:depth]) if len(tokens) >= depth else tuple(tokens)
            groups[key].append(reg)
        return groups

    # Start with depth 1, increase if files are too large
    best_depth = 1
    for depth in range(1, 5):
        groups = group_by_depth(registers, depth)
        max_size = max(len(v) for v in groups.values()) if groups else 0
        if max_size <= max_per_file:
            best_depth = depth
            break
        best_depth = depth

    groups = group_by_depth(registers, best_depth)

    # Now convert to (folder, filename) format
    # Folder = first token, Filename = remaining tokens joined
    result = defaultdict(list)
    for key, regs in groups.items():
        if len(key) == 0:
            folder = "misc"
            filename = "registers"
        elif len(key) == 1:
            folder = key[0]
            filename = "registers"
        else:
            folder = key[0]
            filename = "_".join(key[1:])
        result[(folder, filename)].extend(regs)

    # Merge small groups: if a folder has only one file, flatten it
    # Also merge files with < min_per_file registers into their folder's "misc"
    final_result = defaultdict(list)
    folder_files = defaultdict(list)  # folder -> list of (filename, regs)

    for (folder, filename), regs in result.items():
        folder_files[folder].append((filename, regs))

    for folder, files in folder_files.items():
        if len(files) == 1:
            # Single file in folder - flatten to parent level with combined name
            filename, regs = files[0]
            if filename == "registers":
                # Use folder name as the filename
                final_result[("", folder)].extend(regs)
            else:
                # Combine folder and filename
                combined = f"{folder}_{filename}"
                final_result[("", combined)].extend(regs)
        else:
            # Multiple files - keep the structure but merge tiny ones
            tiny_regs = []
            for filename, regs in files:
                if len(regs) < min_per_file:
                    tiny_regs.extend(regs)
                else:
                    final_result[(folder, filename)].extend(regs)

            if tiny_regs:
                # Merge tiny groups into "misc" or the main file
                if (folder, "registers") in final_result:
                    final_result[(folder, "registers")].extend(tiny_regs)
                else:
                    final_result[(folder, "misc")].extend(tiny_regs)

    # Second pass: flatten any remaining single-file folders
    second_pass = defaultdict(list)
    folder_counts = defaultdict(int)
    for (folder, filename), regs in final_result.items():
        if folder:
            folder_counts[folder] += 1

    for (folder, filename), regs in final_result.items():
        if folder and folder_counts[folder] == 1:
            # Single file in this folder - flatten
            combined = f"{folder}_{filename}" if filename != "misc" else folder
            second_pass[("", combined)].extend(regs)
        else:
            second_pass[(folder, filename)].extend(regs)

    return second_pass


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <html_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    html_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not html_dir.is_dir():
        print(f"Error: {html_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    html_files = list(html_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files")

    modules = defaultdict(list)
    skipped = []

    for filepath in html_files:
        filename = filepath.stem

        if filename.startswith("_") or filename.startswith("mod___"):
            skipped.append(filename)
            continue
        if filename in ("branding", "hh_toc", "hh_search",
                        "am025-versal-aie-ml-register-reference",
                        "am029-versal-aie-ml-v2-register-reference"):
            skipped.append(filename)
            continue

        if "___" not in filename:
            skipped.append(filename)
            continue

        module, reg_name = filename.split("___", 1)
        reg = parse_register_html(filepath)
        if reg:
            reg["_filename"] = reg_name
            modules[module].append(reg)
        else:
            print(f"Warning: Could not parse {filepath}", file=sys.stderr)

    print(f"Skipped {len(skipped)} non-register files")
    print(f"Found {sum(len(v) for v in modules.values())} registers across {len(modules)} modules")

    total_files = 0
    for module, registers in modules.items():
        print(f"\nProcessing {module}: {len(registers)} registers")

        grouping = compute_grouping(registers)
        module_dir = output_dir / module

        for (folder, filename), regs in grouping.items():
            if folder:
                file_dir = module_dir / folder
            else:
                file_dir = module_dir

            file_dir.mkdir(parents=True, exist_ok=True)
            regs.sort(key=lambda r: r.get("offset", ""))

            filepath = file_dir / f"{filename}.txt"
            path_display = f"{folder}/{filename}" if folder else filename

            content_lines = [
                f"# {module.upper()} / {path_display.upper()}",
                f"# {len(regs)} register(s)",
                "#",
                "# Format: NAME @ OFFSET [WIDTH TYPE reset=VALUE] \"DESCRIPTION\"",
                "#   BITS    FIELD_NAME                   TYPE  \"DESCRIPTION\"",
                "",
            ]

            for reg in regs:
                content_lines.append(format_register_compact(reg))
                content_lines.append("")

            with open(filepath, "w") as f:
                f.write("\n".join(content_lines))

        print(f"  Created {len(grouping)} files")
        total_files += len(grouping)

    print(f"\nDone! Created {total_files} files in {output_dir}")


if __name__ == "__main__":
    main()
