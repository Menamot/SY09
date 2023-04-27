def kmeans_dataset(dataset, n_clusters_list, strategies, tries):
    for n_clusters in n_clusters_list:
        for strategy in strategies:
            for rs in range(tries):  # On utilisera `rs` pour fixer le `random_state`
                # <answer>
                inertia = (
                    KMeans(
                        n_clusters=n_clusters,
                        n_init=1,
                        random_state=rs,
                        init=strategy,
                    )
                    .fit(dataset)
                    .inertia_
                )
                # </answer>
                yield rs, strategy, n_clusters, inertia


# <answer>
gen = kmeans_dataset(iris0, [3, 5, 10], ["random", "k-means++"], 100)
# </answer>

df = pd.DataFrame(gen, columns=["seed", "init", "n_clusters", "inertia"])
df = df.astype({
    "seed": "int32",
    "n_clusters": "int32"
})
