import gradio as gr

from graphgen import generate_graph
_TITLE = 'This app generates graphs.'
_DESCRIPTION = 'This app generates graphs based on provided parameters.'

# def map_choice(str: 

def generate(graph_prefix: str,
             N: int,
             c: float,
             k: int,
             Db: int,
             structure_type: str) -> str:
    """Generate graph according to parameters.

    Args:
      N: A number of nodes in the graph.
      c: A number between  0.01-0.999 that defines ratio of structural nodes in the graph.
      Db: An average degree of bridge nodes.

    """
    
    o = 'Parameters: N:%d c:%f Db:%f ->' % (N, c, Db)
    if c < 0.01:
        return 'Ratio must be bigger than 0.01.'
    if c > 0.999:
        return 'Ratio must be smaller than 0.999.'
    print('Received %s' % o)
    generate_graph('../../data', 
                   file_prefix=graph_prefix, 
                   N=int(N), 
                   k=int(k),
                   ratioC=c, 
                   bridge_degree=int(Db))

    o += 'Generated!'
    
    return o

    
genif = gr.Interface(
    generate,
    inputs=[gr.Textbox(value='test', lines=1, max_lines=1, label='File prefix for the graphs.'),
            gr.Number(144, label='N - total number of nodes.'),
            gr.Slider(0.01, 0.99, value=0.75, label='C - ratio of structural nodes'),
            gr.Number(4, label='k - parameter that defines structure size.'),
            gr.Number(8, label='Db - average bridge degree.'),
            gr.Dropdown(value=str, choices=['lattice', 'clique', 'latice and diagonal', 'lattice and all diagonals'])],
    outputs=[gr.Textbox(value='', lines=20, max_lines=20)],
    title=_TITLE,
    description=_DESCRIPTION,
    allow_flagging='never',
)


if __name__=='__main__':
    genif.launch()
