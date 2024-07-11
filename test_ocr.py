from pytesseract import pytesseract 
from PIL import Image
from unidecode import unidecode
import cv2
import json
import os

## Change to your directory
# path_to_tesseract = r"C:\Tesseract-OCR\tesseract.exe"
path_to_tesseract = '/usr/bin/tesseract'
image_path = ['sample_img.jpg','sample_img_checkbox-good.jpg']
region_config_path = 'region_config.json'
json_output_path = 'output.json'


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    pytesseract.tesseract_cmd = path_to_tesseract 
    text = pytesseract.image_to_string(image)
    
    # Save the extracted text to a file
    output_file = 'extracted_text.txt'
    with open(output_file, 'w+', encoding='utf-8') as f:
        f.write(text[:-1])
    print('extracted_text is in ' + os.getcwd() + ' folder.')

def get_text_from_region(img_path, region):
    pytesseract.tesseract_cmd = path_to_tesseract 
    image = cv2.imread(img_path, 0)
    thresh = 255 - cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

    ROI = thresh[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
    data = unidecode(pytesseract.image_to_string(ROI, lang='eng', config='--psm 6'))
    return data


def checkbox_if_present_in_region(img_path, regions):
    # Load the image and crop to the specified region
    image = cv2.imread(img_path)
    # x, y, w, h = regions
    x, y, w, h = tuple(map(int, regions[0].split(', ')))
    region_img = image[y:y+h, x:x+w]

    # Convert the region to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area to remove noise
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    AREA_THRESHOLD = 100
    for c in cnts:
        area = cv2.contourArea(c)
        if area < AREA_THRESHOLD:
            cv2.drawContours(thresh, [c], -1, 0, -1)

    # Repair checkbox horizontal and vertical walls
    repair_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    repair = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, repair_kernel1, iterations=1)
    repair_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, repair_kernel2, iterations=1)

    # Detect checkboxes using shape approximation and aspect ratio filtering
    checkbox_contours = []
    cnts, _ = cv2.findContours(repair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
        aspect_ratio = w_rect / float(h_rect)
        if len(approx) == 4 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2) and (20 < w_rect < 50 and 20 < h_rect < 50):
            checkbox_contours.append((x_rect, y_rect, w_rect, h_rect))

    num_checkboxes = len(checkbox_contours)
    
    if num_checkboxes == 1:
        x_rect, y_rect, w_rect, h_rect = checkbox_contours[0]
        checkbox_roi = repair[y_rect:y_rect + h_rect, x_rect:x_rect + w_rect]
        non_zero_pixels = cv2.countNonZero(checkbox_roi)
        total_pixels = w_rect * h_rect
        filled_ratio = non_zero_pixels / float(total_pixels)
        if filled_ratio > 0.5:  # Adjust threshold as necessary
            return get_text_from_region(img_path, tuple(map(int, regions[1].split(', ')))).rstrip('\n')

    else:
        marked_checkboxes = []
        for index, (x_rect, y_rect, w_rect, h_rect) in enumerate(checkbox_contours, start=1):
            checkbox_roi = repair[y_rect:y_rect + h_rect, x_rect:x_rect + w_rect]
            non_zero_pixels = cv2.countNonZero(checkbox_roi)
            total_pixels = w_rect * h_rect
            filled_ratio = non_zero_pixels / float(total_pixels)

            if filled_ratio > 0.5:  # Adjust threshold as necessary
                marked_checkboxes.append(index)
                
        if len(marked_checkboxes) == 1:
            if marked_checkboxes[0] == 2:
                return get_text_from_region(img_path, tuple(map(int, regions[1].split(', ')))).rstrip('\n')



def set_json_data(img_path, region_config):
    data_dict = {'doctor': {}, 'patient': {'lab_prev': {'wbc': {}}, 'lab_latest': {'wbc': {}}}, 'meds': [], 'allergies': []}
    data_dict['doctor']['name'] = get_text_from_region(img_path[0], tuple(map(int, region_config['doctor']['name'].split(', ')))).rstrip('\n')
    data_dict['doctor']['specialization'] = get_text_from_region(img_path[0], tuple(map(int, region_config['doctor']['specialization'].split(', ')))).rstrip('\n')
    data_dict['doctor']['fellowship'] = get_text_from_region(img_path[0], tuple(map(int, region_config['doctor']['fellowship'].split(', ')))).rstrip('\n')

    data_dict['patient']['name'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['name'].split(', ')))).rstrip('\n')
    data_dict['patient']['birthdate'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['birthdate'].split(', ')))).rstrip('\n')
    data_dict['patient']['abstract'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['abstract'].split(', ')))).replace('\n',' ')
    data_dict['patient']['patient_wt'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['weight'].split(', ')))).replace('\n',' ')
    data_dict['patient']['patient_len'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['length'].split(', ')))).replace('\n',' ')
    data_dict['patient']['patient_hc'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['hc'].split(', ')))).replace('\n',' ')
    data_dict['patient']['impressions'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['impressions'].split(', ')))).replace('\n',' ')

    data_dict['patient']['lab_prev']['date'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['date'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['hb'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['hb'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['hct'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['hct'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['wbc']['value'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['wbc']['value'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['wbc']['lympho'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['wbc']['lympho'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['wbc']['granulo'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['wbc']['granulo'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['wbc']['mono'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['wbc']['mono'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_prev']['platelet'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_prev']['platelet'].split(', ')))).replace('\n',' ')
    
    data_dict['patient']['lab_latest']['date'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['date'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['hb'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['hb'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['hct'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['hct'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['wbc']['value'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['wbc']['value'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['wbc']['lympho'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['wbc']['lympho'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['wbc']['granulo'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['wbc']['granulo'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['wbc']['mono'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['wbc']['mono'].split(', ')))).replace('\n',' ')
    data_dict['patient']['lab_latest']['platelet'] = get_text_from_region(img_path[0], tuple(map(int, region_config['patient']['lab_latest']['platelet'].split(', ')))).replace('\n',' ')


    for item in region_config['meds']:
        drug = "" if None else checkbox_if_present_in_region(img_path[1], item)
        if drug:
            data_dict['meds'].append(drug)

    for item in region_config['allergies']:
        drug = "" if None else checkbox_if_present_in_region(img_path[1], item)
        if drug:
            data_dict['allergies'].append(drug)
    

    json_data = json.dumps(data_dict)

    return json_data


def read_region_config(file_path):
    with open(file_path) as f:
        return json.load(f)
    
def write_json_file(json_data, file_path):
    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(json_data)
    print("Check JSON file " + file_path + " for output...")

    
def main():
    region_config = read_region_config(region_config_path)
    write_json_file(set_json_data(image_path, region_config), json_output_path)
    extract_text_from_image(image_path[0])
    print(set_json_data(image_path, region_config))


main()