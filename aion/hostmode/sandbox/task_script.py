# jail script: read dataset.txt in the working directory, emit the sorted
# unique-value report to stdout. stdout is DATA for the agent to persist.
counts = {}
with open('dataset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        val = line.split(',', 1)[1]
        counts[val] = counts.get(val, 0) + 1
for val in sorted(counts):
    print(val + ' ' + str(counts[val]))
