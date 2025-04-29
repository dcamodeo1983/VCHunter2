if founder_embedding_2d is not None:
    founder_x, founder_y = founder_embedding_2d
    fig.add_scatter(
        x=[founder_x],
        y=[founder_y],
        mode="markers",
        marker=dict(
            symbol="star",
            size=22,
            color="gold",
            line=dict(color="black", width=2)
        ),
        name="Founder Idea"
    )
