import torch
from diffusers import StableDiffusionInpaintPipeline, PaintByExamplePipeline, AutoPipelineForInpainting
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
from datetime import datetime

# Load and preprocess image
def load_image(image_path):
    return Image.open(image_path).convert('RGB')

# Create mask for black areas
def create_mask(image):
    return Image.fromarray((np.array(image) == 0).all(axis=2).astype(np.uint8) * 255)

# Fill black areas
def fill_black_areas(pipe, image, mask, prompt, negative_prompt):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask).images[0]

# Update the alternative model function
def fill_black_areas_alt(pipe, image, mask, prompt, negative_prompt):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask).images[0]

# Update the Paint-by-Example function
def fill_black_areas_pbe(pipe, image, mask, example_image):
    return pipe(image=image, mask_image=mask, example_image=example_image).images[0]

# Display original and filled images side by side
def display_results(original_image, filled_image1, filled_image2, filled_image3, output_base_path):
    plt.style.use('default')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
    
    ax1.imshow(original_image)
    ax1.set_title('A', fontweight='bold', loc='left')
    ax1.axis('off')
    
    ax2.imshow(filled_image1)
    ax2.set_title('B', fontweight='bold', loc='left')
    ax2.axis('off')
    
    ax3.imshow(filled_image2)
    ax3.set_title('C', fontweight='bold', loc='left')
    ax3.axis('off')
    
    ax4.imshow(filled_image3)
    ax4.set_title('D', fontweight='bold', loc='left')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='w', marker='s', markersize=15, markerfacecolor='gray', label='Original'),
        plt.Line2D([0], [0], color='w', marker='s', markersize=15, markerfacecolor='lightblue', label='Stable Diffusion v2'),
        plt.Line2D([0], [0], color='w', marker='s', markersize=15, markerfacecolor='lightgreen', label='Stable Diffusion v1'),
        plt.Line2D([0], [0], color='w', marker='s', markersize=15, markerfacecolor='lightsalmon', label='Paint-by-Example')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make room for legend
    
    # Save as PNG
    plt.savefig(f"{output_base_path}.png", dpi=300, bbox_inches='tight')
    # Save as SVG
    plt.savefig(f"{output_base_path}.svg", format='svg', bbox_inches='tight')
    
    # Save the figure object
    with open(f"{output_base_path}.pkl", 'wb') as file:
        pickle.dump(fig, file)
    
    plt.close()

# Main function
def main():
    # Set the directory path
    image_dir = '/workspace/ImageRecognition/1_data/patches_120x120/patches_cutoff_black_imputed_0.7/Inside'

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    # Select a random image file
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)

    # Load pretrained models
    pipe1 = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe2 = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe3 = PaintByExamplePipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=torch.float16
    ).to("cuda")

    # Load and process image
    image = load_image(image_path)
    mask = create_mask(image)

    # For Paint-by-Example, we need an example image. Let's use another random image from the same directory.
    example_image_file = random.choice([f for f in image_files if f != random_image])
    example_image_path = os.path.join(image_dir, example_image_file)
    example_image = load_image(example_image_path)

    # Create a new subdirectory with datetime as name
    now = datetime.now()
    datetime_string = now.strftime("%Y%m%d_%H%M%S")
    output_dir = '/workspace/ImageRecognition/5_results/generative_fill'
    output_subdir = os.path.join(output_dir, datetime_string)
    os.makedirs(output_subdir, exist_ok=True)

    # Define prompt and negative prompt
    prompt = "use neutral colors from hematoxylin and eosin staining that best matches the rest of the image"
    negative_prompt = "mismatching colors with rest of image, higher resolution than the rest of the image"

    # Fill black areas with all models
    filled_image1 = fill_black_areas(pipe1, image, mask, prompt, negative_prompt)
    filled_image2 = fill_black_areas_alt(pipe2, image, mask, prompt, negative_prompt)
    filled_image3 = fill_black_areas_pbe(pipe3, image, mask, example_image)

    # Save results
    output_base_path = os.path.join(output_subdir, f'filled_{os.path.splitext(random_image)[0]}')
    
    # Display and save figure
    display_results(image, filled_image1, filled_image2, filled_image3, output_base_path)
    
    print(f"Processed image: {random_image}")
    print(f"Example image used: {example_image_file}")
    print(f"Saved results in directory: {output_subdir}")
    print(f"Saved filled images to: {output_base_path}.png and {output_base_path}.svg")
    print(f"Saved editable plot object to: {output_base_path}.pkl")

if __name__ == '__main__':
    main()
