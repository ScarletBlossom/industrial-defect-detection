import gradio as gr
import torch
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.predictor import DefectDetector
from src.config import Config


def initialize_detector():
    try:
        detector = DefectDetector(checkpoint_path='checkpoints/best_model.pth')
        print("Model loaded successfully!")
        return detector
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


detector = initialize_detector()


def predict_defect(image):
    if detector is None:
        return "Model not loaded. Please train the model first.", None
    
    if image is None:
        return "Please upload an image.", None
    
    result = detector.predict_image(image)
    
    class_name = result['class']
    confidence = result['confidence'] * 100
    
    prob_normal = result['probabilities'][0] * 100
    prob_defect = result['probabilities'][1] * 100
    
    output_text = f"**Prediction: {class_name}**\n\n"
    output_text += f"Confidence: {confidence:.2f}%\n\n"
    output_text += f"Probabilities:\n"
    output_text += f"  - Normal: {prob_normal:.2f}%\n"
    output_text += f"  - Defect: {prob_defect:.2f}%"
    
    return output_text, class_name


def create_interface():
    with gr.Blocks(title="Industrial Defect Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Industrial Defect Detection System")
        gr.Markdown("Upload a product surface image to detect whether it has defects.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image", height=300)
                with gr.Row():
                    predict_btn = gr.Button("Detect Defect", variant="primary")
                    clear_btn = gr.Button("Clear")
            
            with gr.Column():
                output_label = gr.Markdown(label="Prediction Result")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        predict_btn.click(predict_defect, inputs=[image_input], outputs=[output_label, status_text])
        clear_btn.click(lambda: (None, ""), outputs=[image_input, output_label])
        
        gr.Markdown("---")
        gr.Markdown("### Model Information")
        gr.Markdown(f"- **Model**: ResNet18 (pretrained on ImageNet)")
        gr.Markdown(f"- **Classes**: Normal, Defect")
        gr.Markdown(f"- **Device**: {Config.DEVICE}")
        
        gr.Markdown("---")
        gr.Markdown("*Upload a product surface image to test the defect detection system.*")
    
    return demo


if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)