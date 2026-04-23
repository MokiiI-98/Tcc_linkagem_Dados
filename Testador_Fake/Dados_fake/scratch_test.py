import pandas as pd
import recordlinkage

df_a = pd.read_csv('sinasc_100.csv', sep=';', dtype=str).reset_index()
df_b = pd.read_csv('sim_100.csv', sep=';', dtype=str).reset_index()
true_matches = pd.read_csv('matches_100.csv', sep=';', dtype=str)

df_a["id_sinasc"] = df_a.index
df_b["id_sim"] = df_b.index

map_sim = {old: new for new, old in enumerate(df_b.index)}
map_sinasc = {old: new for new, old in enumerate(df_a.index)}
df_matches_idx = true_matches.iloc[:, :2].copy()
df_matches_idx = df_matches_idx.astype(int)
df_matches_idx.columns = ["sinasc_index", "sim_index"]
df_matches_idx["sinasc_index"] = df_matches_idx["sinasc_index"].map(map_sinasc)
df_matches_idx["sim_index"] = df_matches_idx["sim_index"].map(map_sim)
true_matches = pd.MultiIndex.from_frame(df_matches_idx[["sinasc_index", "sim_index"]])

indexer = recordlinkage.Index()
# Let's use sortedneighbourhood on NOME
indexer.sortedneighbourhood("NOME", "NOME", window=5)
indexer.sortedneighbourhood("NOMEMAE", "NOMEMAE", window=5)
indexer.block(["DTNASC"])
indexer.block(["CODMUNRES", "SEXO"])

links = indexer.index(df_a, df_b)
links = links.drop_duplicates()

inter = true_matches.intersection(links)
print(f"True matches in candidate links: {len(inter)} out of {len(true_matches)}")
print(f"Total candidate links: {len(links)}")
