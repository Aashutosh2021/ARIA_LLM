from model.mask import causal_mask

mask = causal_mask(5)

print(mask.shape)

print(mask[0][0])