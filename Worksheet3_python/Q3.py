org1 = ["ACGTTTCA", "AGGCCTTA", "AAAACCTG"]
org2 = ["AGCTTTGA", "GCCGGAAT", "GCTACTGA"]
threshold = float(input("Enter the threshold you want to give: "))

def similarity(seq1, seq2):
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)

similar_pairs = [(seq1, seq2) for seq1 in org1 for seq2 in org2 if similarity(seq1, seq2) > threshold]
print(similar_pairs)
