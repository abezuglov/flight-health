"""List every class/function a pickle would import, without loading it."""
import pickletools, sys, collections

path = sys.argv[1]
found = collections.Counter()
recent = []  # last string pushes, for STACK_GLOBAL
reduce_ops = 0
with open(path, "rb") as f:
    for op, arg, pos in pickletools.genops(f):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            recent = (recent + [arg])[-2:]
        elif op.name == "GLOBAL":
            found[arg.replace(" ", ".")] += 1
        elif op.name == "STACK_GLOBAL" and len(recent) == 2:
            found[f"{recent[0]}.{recent[1]}"] += 1
        elif op.name in ("REDUCE", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ"):
            reduce_ops += 1
for name, n in sorted(found.items()):
    print(f"{n:>8}  {name}")
print("object-building ops:", reduce_ops)
