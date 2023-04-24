import sys
import json

def main():
    if len(sys.argv) != 4:
        return 1
    file_path = sys.argv[1]
    extract_params = sys.argv[2].split(",")
    limit = int(sys.argv[3])
    data = []
    with open(file_path, "r", encoding="latin-1") as f:
        f = json.load(f)
        for _ in range(len(f) if len(f) < limit else limit):
            x, y = f[_].get(extract_params[0], None), f[_].get(extract_params[1], None)
            data.append([x, y])

    with open("extracted.json", "w") as of:
        json.dump(data, of)

if __name__ == "__main__":
    main()