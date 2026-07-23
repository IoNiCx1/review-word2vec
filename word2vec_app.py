import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Word2Vec Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Word2Vec Explorer")
st.caption("Amazon Fine Food Reviews — Skip-gram · 300 dims · 37,985 words")

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    return Word2Vec.load("word2vec_amazon.model")

with st.spinner("Loading model..."):
    model = load_model()

st.success(f"Model loaded — {len(model.wv):,} words · {model.vector_size} dimensions")
st.divider()

# ── Sidebar navigation ────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Choose a section",
    ["Similar Words", "Word Analogies", "Similarity Heatmap", "t-SNE Map"]
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Similar Words
# ══════════════════════════════════════════════════════════════════════════════
if section == "Similar Words":
    st.header("🔍 Find Similar Words")
    st.write("Type any word and see which words the model thinks are closest to it.")

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Enter a word", value="delicious")
    with col2:
        topn = st.slider("How many results", 5, 20, 10)

    if query:
        if query.lower() not in model.wv:
            st.error(f"'{query}' is not in the vocabulary. Try another word.")
        else:
            results = model.wv.most_similar(query.lower(), topn=topn)

            # Table
            st.subheader(f"Top {topn} words closest to **'{query}'**")
            cols = st.columns([3, 2, 5])
            cols[0].markdown("**Word**")
            cols[1].markdown("**Similarity**")
            cols[2].markdown("**Score bar**")

            for word, score in results:
                c1, c2, c3 = st.columns([3, 2, 5])
                c1.write(word)
                c2.write(f"{score:.4f}")
                c3.progress(float(score))

            # Bar chart
            st.subheader("Chart")
            fig, ax = plt.subplots(figsize=(8, 4))
            words  = [w for w, _ in results]
            scores = [s for _, s in results]
            ax.barh(words[::-1], scores[::-1], color="steelblue")
            ax.set_xlabel("Cosine Similarity")
            ax.set_xlim(0, 1)
            ax.set_title(f"Most similar to '{query}'", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Word Analogies
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Word Analogies":
    st.header("🧮 Word Analogies")
    st.write("Vector arithmetic: **A + B − C ≈ ?**  Classic example: `king + woman − man ≈ queen`")

    col1, col2, col3 = st.columns(3)
    with col1:
        pos1 = st.text_input("Add word 1 (+)", value="sweet")
    with col2:
        pos2 = st.text_input("Add word 2 (+)", value="coffee")
    with col3:
        neg1 = st.text_input("Subtract word (−)", value="bitter")

    if st.button("Run analogy"):
        words = [pos1.lower(), pos2.lower(), neg1.lower()]
        missing = [w for w in words if w not in model.wv]

        if missing:
            st.error(f"Not in vocabulary: {missing}")
        else:
            results = model.wv.most_similar(
                positive=[pos1.lower(), pos2.lower()],
                negative=[neg1.lower()],
                topn=8
            )
            st.subheader(f"`{pos1} + {pos2} − {neg1}` ≈")
            for i, (word, score) in enumerate(results, 1):
                st.write(f"**{i}.** `{word}` — {score:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Similarity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Similarity Heatmap":
    st.header("🌡️ Cosine Similarity Heatmap")
    st.write("Enter a comma-separated list of words to compare all pairs at once.")

    default = "delicious, tasty, awful, terrible, coffee, tea, shipping, fresh, stale"
    raw = st.text_area("Words to compare (comma-separated)", value=default, height=80)

    if st.button("Draw heatmap"):
        # Parse input
        input_words = [w.strip().lower() for w in raw.split(",") if w.strip()]
        in_vocab    = [w for w in input_words if w in model.wv]
        skipped     = [w for w in input_words if w not in model.wv]

        if skipped:
            st.warning(f"Skipped (not in vocab): {skipped}")

        if len(in_vocab) < 2:
            st.error("Need at least 2 words that are in the vocabulary.")
        else:
            # Build similarity matrix
            n   = len(in_vocab)
            mat = np.zeros((n, n))
            for i, w1 in enumerate(in_vocab):
                for j, w2 in enumerate(in_vocab):
                    mat[i, j] = model.wv.similarity(w1, w2)

            fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
            sns.heatmap(mat,
                        xticklabels=in_vocab, yticklabels=in_vocab,
                        annot=True, fmt=".2f", cmap="RdYlGn",
                        vmin=-0.2, vmax=1.0, linewidths=0.5, ax=ax)
            ax.set_title("Pairwise Cosine Similarity", fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — t-SNE Map
# ══════════════════════════════════════════════════════════════════════════════
elif section == "t-SNE Map":
    st.header("🗺️ t-SNE Word Map")
    st.write("Projects 300-dimensional word vectors down to 2D. Similar words cluster together.")

    # Word groups
    default_groups = {
        "Positive taste"  : "delicious, tasty, yummy, flavorful, savory",
        "Negative quality": "terrible, awful, horrible, disgusting, bland",
        "Beverages"       : "coffee, tea, juice, water, soda",
        "Snacks"          : "chip, cookie, cracker, pretzel, popcorn",
        "Shipping"        : "shipping, delivery, package, arrived, damaged",
    }

    st.subheader("Edit word groups")
    group_inputs = {}
    for group, words in default_groups.items():
        group_inputs[group] = st.text_input(group, value=words)

    if st.button("Generate t-SNE map"):
        words_to_plot, labels_to_plot = [], []
        for group, raw in group_inputs.items():
            for word in [w.strip().lower() for w in raw.split(",")]:
                if word and word in model.wv:
                    words_to_plot.append(word)
                    labels_to_plot.append(group)

        if len(words_to_plot) < 5:
            st.error("Need at least 5 words in the vocabulary across all groups.")
        else:
            with st.spinner("Running PCA + t-SNE (may take ~30 seconds)..."):
                vectors = np.array([model.wv[w] for w in words_to_plot])

                # PCA 300D → 50D first (speeds up t-SNE significantly)
                n_pca       = min(50, len(words_to_plot) - 1)
                vectors_50d = PCA(n_components=n_pca, random_state=42).fit_transform(vectors)

                # t-SNE 50D → 2D
                vectors_2d = TSNE(
                    n_components  = 2,
                    perplexity    = min(15, len(words_to_plot) // 2),
                    n_iter        = 1000,
                    random_state  = 42,
                    learning_rate = "auto",
                    init          = "pca",
                ).fit_transform(vectors_50d)

            # Plot
            unique_groups = list(group_inputs.keys())
            colors        = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
            color_map     = dict(zip(unique_groups, colors))

            fig, ax = plt.subplots(figsize=(12, 8))
            for i, (word, group) in enumerate(zip(words_to_plot, labels_to_plot)):
                x, y = vectors_2d[i]
                c    = color_map[group]
                ax.scatter(x, y, color=c, s=100, zorder=3)
                ax.annotate(word, (x, y), xytext=(5, 3),
                            textcoords="offset points",
                            fontsize=9, color=c, fontweight="bold")

            # Legend
            from matplotlib.lines import Line2D
            ax.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[g], markersize=10, label=g)
                    for g in unique_groups
                ],
                loc="upper left", fontsize=10, framealpha=0.8
            )
            ax.set_title("Word2Vec Embeddings — t-SNE", fontsize=14, fontweight="bold")
            ax.grid(linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(f"Plotted {len(words_to_plot)} words across {len(unique_groups)} groups.")