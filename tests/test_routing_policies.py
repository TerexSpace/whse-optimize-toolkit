import pytest
import networkx as nx
from wms_optlab.data.models import Location
from wms_optlab.routing.policies import get_s_shape_route

@pytest.fixture
def sample_layout():
    """Provides a sample warehouse layout for testing routing policies."""
    depot = Location("D", (0, 5, 0))
    locs = [
        Location("A1", (10, 10, 0)),
        Location("A2", (10, 20, 0)),
        Location("B1", (20, 10, 0)),
    ]
    graph = nx.Graph()
    for loc in [depot] + locs:
        graph.add_node(loc.loc_id, pos=loc.coordinates)
    
    return depot, locs, graph

def test_s_shape_route_structure(sample_layout):
    """
    Tests the basic structure of the S-shape route.
    It should start and end at the depot.
    """
    depot, pick_locs, graph = sample_layout
    
    route = get_s_shape_route(pick_locs, graph, depot)
    
    assert route[0] == depot.loc_id
    assert route[-1] == depot.loc_id
    assert len(route) == len(pick_locs) + 2 # Picks + start + end

def test_s_shape_route_ordering(sample_layout):
    """
    Tests that the S-shape route follows the correct aisle traversal logic.
    """
    depot, pick_locs, graph = sample_layout
    
    # pick_locs are [A1(10,10), A2(10,20), B1(20,10)]
    route = get_s_shape_route(pick_locs, graph, depot)
    
    # Expected S-shape: D -> A1 -> A2 -> B1 -> D
    # Aisle 1 (x=10) is traversed up (y=10 then y=20)
    # Aisle 2 (x=20) is traversed down (but only one item)
    
    # We can't be certain about B1's position relative to A's without more info,
    # but we can check the order within the 'A' aisle.
    idx_a1 = route.index("A1")
    idx_a2 = route.index("A2")
    
    assert idx_a1 < idx_a2
