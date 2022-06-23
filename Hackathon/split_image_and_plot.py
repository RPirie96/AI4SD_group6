graphene_h = np.vsplit(graphene[1], 4)
plt.imshow(graphene_h[0])


def split_to_patches(array, split_h, split_v):
    h_split = np.hsplit(array, split_h)
    values = []
    for x in h_split:
        for y in np.vsplit(x, split_v):
            values.append(y)

    return values

a = split_to_patches(graphene[0], 4, 4)
fig, ax = plt.subplots(4, 4)
for i, x in enumerate(ax.ravel()):
    x.imshow(a[i])