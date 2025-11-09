# Quick test to understand trail ordering
trail = [[10, 5], [11, 5], [12, 5]]

print("Trail:", trail)
print("Head (first or last?):", trail[0], "or", trail[-1])

# If we moved from [10,5] -> [11,5] -> [12,5]
# Then head should be [12, 5] (last element)
# But code uses trail[0] as head in some places

