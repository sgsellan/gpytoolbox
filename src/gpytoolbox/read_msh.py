import warnings
from typing import Dict, List, Tuple

SUPPORTED_MSH_VERSIONS = ["4.1", ]


def read_msh(file: str) -> Dict[Tuple[int, int],
                                Tuple[List[Tuple[int, int]],
                                      Tuple[List[int],
                                            List[float],
                                            List[float]],
                                      Tuple[List[int],
                                            List[List[int]],
                                            List[List[int]]],
                                      str]
                                ]:
    """Reads a gmsh mesh from .msh file.

    Only the following sections are currently supported and will be imported: $Entities, $Nodes, $Elements

    Parameters
    -------
    file : string
        the path the mesh will be read from

    Returns
    ----------
    mesh : Dict
        Dict containing all entities, nodes and elements in the following structure
        Adapted from the structure in https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/copy_mesh.py
        {(dim, tag): ([(boundary_dim0, boundary_tag0), (boundary_dim1, boundary_tag1), ...] ,
                      ([node0_tag, node1_tag, ...],
                       [node0_x, node0_y, node_z0, node1_x, node1_y, node_z1, ...],
                       [node0_parametric_coords, node1_parametric_coords, ...]),
                      ([element_type0, element_type1, ...],
                       [[element0_tag, element1_tag, ...],   <-- for element_type0
                        [element0_tag, element1_tag, ...],   <-- for element_type1
                        ...],
                       [[e0_node_tag0, e0_node_tag1, ..., e0_node_tagN, e1_node_tag0, ...], <-- for element_type0
                        [e0_node_tag0, e0_node_tag1, ..., e0_node_tagN, e1_node_tag0, ...], <-- for element_type1
                        ...]),
                      entity_name),
         (dim, tag): ...}

        Can be obtained from gmsh directly via
        mesh = {}
        for dim, tag in gmsh.model.getEntities():
            mesh[(dim, tag)] = (gmsh.model.getBoundary([(dim, tag)]),
                                gmsh.model.mesh.getNodes(dim, tag),
                                gmsh.model.mesh.getElements(dim, tag),
                                gmsh.model.get_entity_name(dim, tag))

    Examples
    --------
    ```python
    # Read a mesh in gmsh .msh format
    mesh = gpytoolbox.read_msh('mesh.msh')
    ```

    """
    # get (all) helper functions
    helpers = globals()

    # empty output
    mesh = {}

    # read file
    with open(file, 'r') as f:
        lines = f.read().splitlines()

    # number of lines and current line
    n_lines = len(lines)
    cur_line = 0

    # loop over all sections
    while cur_line < n_lines:
        cur_section: str = lines[cur_line]

        # current line should always start with a new section
        if not cur_section.startswith("$"):
            raise ValueError(f"Unsupported file format, line {cur_line} does not start with a new section.")

        # Call appropriate helper function to read this section
        try:
            mesh, cur_line = helpers[cur_section.replace("$", "_")](mesh, lines, cur_line)
        # unsupported section
        except KeyError:
            warnings.warn(f"Section {cur_section} was not imported.")
            # jump to next section
            cur_line = _next_section(lines, cur_line)

    return mesh


def _MeshFormat(mesh: Dict, lines: List[str], cur_line: int) -> Tuple[Dict, int]:
    """
    Check compatible mesh format.

    Parameters
    ----------
    mesh : Dict
        Output dict to store the data in
    lines : List[str]
        List of file lines
    cur_line : int
        Current line

    Returns
    -------
    mesh : Dict
        Output dict to store the data in
    cur_line : int
        Next line after this section

    """
    # current section
    cur_section = lines[cur_line]

    # first line of the block
    cur_line += 1

    # get .msh file version
    version, filetype, _ = lines[cur_line].split(' ')

    # check msh version
    if not version in SUPPORTED_MSH_VERSIONS:
        raise ValueError(f"Incompatible MSH version ({version}), supported are ({', '.join(SUPPORTED_MSH_VERSIONS)})")
    # check ascii mode
    if not filetype == '0':
        raise ValueError(f"Incompatible MSH file format, only ASCII mode is supported currently.")

    # end of block ($MeshFormat is always 3 lines long)
    cur_line = cur_line + 1

    # check current block end
    _check_section_end(lines, cur_line, cur_section)

    # increase cur_line by one to point to the next starting block
    return mesh, cur_line + 1


def _next_section(lines: List[str], cur_line: int) -> int:
    """
    Find starting line of next section.

    Parameters
    ----------
    lines : List[str]
        List of file lines
    cur_line : int
        Current line

    Returns
    -------
    cur_line : int
        Next line after the current section

    """
    # current section
    cur_section = lines[cur_line]

    # first line of the block
    cur_line += 1

    # check if line
    while not lines[cur_line].startswith("$"):
        cur_line += 1

    # check current block end
    _check_section_end(lines, cur_line, cur_section)

    return cur_line + 1


def _check_section_end(lines: List[str], cur_line: int, cur_section: str) -> None:
    """
    Find starting line of next section.

    Parameters
    ----------
    lines : List[str]
        List of file lines
    cur_line : int
        Current line
    cur_section: str
        Current section to check end with

    """
    # check if start and end of the sections match
    if not lines[cur_line] == cur_section[0] + "End" + cur_section[1:]:
        raise ValueError(f"Start {cur_section} and End {lines[cur_line]} of section do not match!")

    return


read_msh(r"../../test_msh.msh")
