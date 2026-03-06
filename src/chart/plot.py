import plotly.graph_objects as go
import numpy as np

def visualize_concept_flow(matrix: 'ConceptMatrix', title: str = "Flujo 3D SPEKTRA"):
    """
    Renderiza un grafo 3D de los nodos existentes en la Matrix y sus conexiones.
    """
    node_coords = []
    node_names = []
    node_energies = []
    
    edge_x = []
    edge_y = []
    edge_z = []
    
    # 1. Extraer Nodos
    for coords, node in matrix._node_storage.items():
        node_coords.append(coords)
        node_names.append(node.name)
        # Usamos la madurez como proxy de tamaño si no hay activación actual
        node_energies.append(getattr(node, 'last_activation_energy', 0.5) * 20)
        
        # 2. Extraer Punteros (Edges)
        for target_idx, strength in node.pointers.items():
            if target_idx in matrix._node_storage:
                target_node = matrix._node_storage[target_idx]
                target_coords = target_node.index
                
                edge_x.extend([coords[0], target_coords[0], None])
                edge_y.extend([coords[1], target_coords[1], None])
                edge_z.extend([coords[2], target_coords[2], None])

    # Crear Trazado de Conexiones
    edges_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Crear Trazado de Nodos
    node_coords_array = np.array(node_coords)
    nodes_trace = go.Scatter3d(
        x=node_coords_array[:, 0],
        y=node_coords_array[:, 1],
        z=node_coords_array[:, 2],
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=node_energies,
            color=node_energies,
            colorscale='Viridis',
            line_width=2
        ),
        text=node_names,
        textposition="top center",
        hoverinfo='text'
    )

    # Configurar Layout
    layout = go.Layout(
        title=title,
        template="plotly_dark",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        margin=dict(b=0, l=0, r=0, t=40)
    )

    fig = go.Figure(data=[edges_trace, nodes_trace], layout=layout)
    fig.show()