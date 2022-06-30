graphene_h = np.vsplit(graphene[1], 4)
plt.imshow(graphene_h[0])


def split_to_patches(array, split_h, split_v):
    v_split = np.vsplit(array, split_v)
    values = []
    for x in v_split:
        for y in np.hsplit(x, split_h):
            values.append(y)

    return values

a = split_to_patches(graphene[0], 4, 4)
fig, ax = plt.subplots(4, 4)
for i, x in enumerate(ax.ravel()):
    x.imshow(a[i])