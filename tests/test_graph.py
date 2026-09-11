import numpy as np

from amortized_mcmc import build_graph


def test_delaunay_graph_has_edges():
    graph = build_graph(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]), method="delaunay")
    assert graph.n_edges > 0
    assert graph.senders.shape == graph.receivers.shape
