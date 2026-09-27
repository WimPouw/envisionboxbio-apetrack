"""The three components to cite (README.md, section Citation, without markdown). Printed on import; set the
environment variable APETRACK_QUIET=1 to switch that off."""
CITATIONS = (
    'OWLv2: Minderer, M., Gritsenko, A., & Houlsby, N. (2023). Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36, 72983–73007. https://doi.org/10.52202/075280-3191',
    'ByteTrack: Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., ... & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. In European Conference on Computer Vision (pp. 1–21). Springer Nature Switzerland. https://doi.org/10.1007/978-3-031-20047-2_1',
    'OpenApePose: Desai, N., Bala, P., Richardson, R., Raper, J., Zimmermann, J., & Hayden, B. (2023). OpenApePose, a database of annotated ape photographs for pose estimation. eLife, 12, RP86873. https://doi.org/10.7554/eLife.86873.3',
)


def print_citations():
    print("envisionboxbio-apetrack builds on three components. Please cite them:")
    for c in CITATIONS:
        print("  - " + c)
