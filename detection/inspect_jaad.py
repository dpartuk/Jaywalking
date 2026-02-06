import os
import xml.etree.ElementTree as ET

# Point this to your main annotations folder
ANNOTATIONS_PATH = "./JAAD/annotations" 

def inspect_one_file():
    # Find the first XML file
    files = [f for f in os.listdir(ANNOTATIONS_PATH) if f.endswith('.xml')]
    if not files:
        print("❌ No XML files found in ./JAAD/annotations")
        print("   -> Check if your folder name is correct.")
        return

    target_file = files[0]
    xml_path = os.path.join(ANNOTATIONS_PATH, target_file)
    print(f"🔍 Inspecting: {target_file}...\n")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    crossing_found = False
    attributes_found = []

    for track in root.findall('track'):
        if track.attrib['label'] == 'pedestrian':
            for box in track.findall('box'):
                # Look for attributes inside the box
                for attr in box.findall('attribute'):
                    name = attr.attrib.get('name')
                    attributes_found.append(name)
                    if name == 'crossing':
                        crossing_found = True
                        val = attr.text
                        print(f"✅ FOUND 'crossing' attribute! (Sample value: {val})")
                        print("   -> We can use the 'annotations' folder directly.")
                        return

    if not crossing_found:
        print("⚠️ 'crossing' attribute NOT found in the main XMLs.")
        print(f"   -> Found these attributes instead: {list(set(attributes_found))}")
        print("   -> We likely need to merge data from 'annotations_attributes'.")

if __name__ == "__main__":
    inspect_one_file()