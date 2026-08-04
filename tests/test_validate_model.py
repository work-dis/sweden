import numpy as np
import sys

sys.path.insert(0, "scripts")


def test_auc_roc_perfect_and_tied():
    from validate_model import auc_roc
    assert auc_roc(np.array([0.8, 0.9]), np.array([0.1, 0.2])) == 1.0
    assert auc_roc(np.array([0.5]), np.array([0.5])) == 0.5


def test_precision_at_k():
    from validate_model import precision_at_k
    precision, k = precision_at_k(np.array([0.9, 0.8]), np.array([0.85, 0.1]))
    assert k == 2
    assert precision == 0.5


def test_precision_at_k_treats_boundary_ties_neutrally():
    from validate_model import precision_at_k
    precision, k = precision_at_k(np.array([0.5]), np.full(9, 0.5))
    assert k == 1
    assert precision == 0.1


def test_spatial_deduplication_prefers_precise_record():
    from validate_model import Occurrence, deduplicate_spatially
    records = [
        Occurrence(59.33000, 18.06000, 500),
        Occurrence(59.33001, 18.06001, 20),
        Occurrence(60.0, 18.0, 100),
    ]
    result = deduplicate_spatially(records)
    assert len(result) == 2
    assert any(record.uncertainty_m == 20 for record in result)


def test_habitat_score_uses_image_dimensions():
    from validate_model import habitat_score
    image = np.full((10, 20), 255, dtype=np.uint8)
    assert habitat_score(62.0, 16.0, image) == 100.0
    assert habitat_score(10.0, 16.0, image) is None


def test_parse_taxon_ids():
    from validate_model import parse_taxon_ids
    parsed = parse_taxon_ids("1,2,3,4,5")
    assert list(parsed.values()) == [1, 2, 3, 4, 5]
