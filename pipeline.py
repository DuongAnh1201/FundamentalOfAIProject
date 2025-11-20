from graphviz import Digraph

def create_pipeline_diagram():
    dot = Digraph('EmotionPipeline', comment='Real-Time Emotion Detection Pipeline')
    dot.attr(rankdir='TB', size='10')
    
    # Define Nodes
    dot.node('Input', '🎥 Webcam Input Frame', shape='ellipse', style='filled', color='lightgrey')
    dot.node('Output', '🖥️ Final Labeled Frame', shape='ellipse', style='filled', color='lightgrey')

    # Stage 1 Subgraph
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='filled', color='lightblue', label='Stage 1: Face Detection')
        c.node('YOLO', '🚀 YOLOv11 Face Detector', shape='box', style='filled', color='white')
        c.node('Box', '📍 Bounding Box Coords', shape='parallelogram', style='filled', color='white')
        c.edge('YOLO', 'Box')

    # Preprocessing Subgraph
    with dot.subgraph(name='cluster_1') as c:
        c.attr(style='filled', color='lightyellow', label='Preprocessing')
        c.node('Crop', '✂️ Crop Face Region', shape='box', style='filled', color='white')
        c.node('Resize', '🔄 Resize to 224x224', shape='box', style='filled', color='white')
        c.node('Norm', '🎨 Normalize (RGB)', shape='box', style='filled', color='white')
        c.edge('Crop', 'Resize')
        c.edge('Resize', 'Norm')

    # Stage 2a Subgraph
    with dot.subgraph(name='cluster_2') as c:
        c.attr(style='filled', color='lightgreen', label='Stage 2a: Feature Extraction')
        c.node('ViT', '👁️ ViT-B/16 Model', shape='box', style='filled', color='white')
        c.node('Vector', '🔢 Feature Vector (768)', shape='parallelogram', style='filled', color='white')
        c.edge('ViT', 'Vector')

    # Stage 2b Subgraph
    with dot.subgraph(name='cluster_3') as c:
        c.attr(style='filled', color='mistyrose', label='Stage 2b: Classification')
        c.node('LogReg', '🧠 Custom Logistic Regression', shape='box', style='filled', color='white')
        c.node('Softmax', '📊 Softmax Activation', shape='box', style='filled', color='white')
        c.node('Result', '😊 Emotion Label', shape='parallelogram', style='filled', color='white')
        c.edge('LogReg', 'Softmax')
        c.edge('Softmax', 'Result')

    # Main Connections
    dot.edge('Input', 'YOLO')
    dot.edge('Input', 'Crop')
    dot.edge('Box', 'Crop')
    dot.edge('Norm', 'ViT')
    dot.edge('Vector', 'LogReg')
    dot.edge('Result', 'Output')

    # Render
    output_path = dot.render('pipeline_architecture', format='png', cleanup=True)
    print(f"Diagram saved to: {output_path}")

if __name__ == '__main__':
    try:
        create_pipeline_diagram()
    except Exception as e:
        print("Error: Could not generate image.")
        print("Make sure you have Graphviz installed: https://graphviz.org/download/")
        print(f"Details: {e}")