from src.utils import read_magic, write_magic

path = read_magic("rsc/jsonl_single_line/path.txt")

result = []
for item in path:
    try:
        x = list(read_magic(item.replace("/home/work/user/SFT/data/Midm_v2.0_2nd/jsonl_format-final", "rsc/jsonl_single_line")))[0]
    except: x = {}
    result.append(x)

write_magic("single_lines.jsonl", result)
