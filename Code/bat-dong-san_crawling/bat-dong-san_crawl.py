from selenium import webdriver
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import pandas as pd
from time import sleep
import random 

def get_page_list(num_pages, base_url):
    url_list = []

    for i in range(num_pages):
        if i == 0:
            url = base_url
            url_list.append(url)
        else:
            url = base_url + "/p" + str(i+1)
            url_list.append(url)
            
    return url_list

def get_information(driver, accoms, feature_list, address_list, publishing_date_list, expiry_date_list, news_category_list):
    for accom in accoms:
        driver.get(accom['URL'])
        sleep(random.randint(3, 7))
    
        feature_elems = driver.find_elements(By.CLASS_NAME , "re__pr-specs-content-item")
        temp = []
        for elem in feature_elems:
            temp.append(elem.text) 
        temp_combined = ',\t'.join(temp)
        feature_list.append(temp_combined) 

        address_elems = driver.find_elements(By.XPATH, "//span[contains(@class, \'re__pr-short-description js__pr-address\')]")
        for elem in address_elems:
            address_list.append(elem.text)
            
        date_elems = driver.find_elements(By.XPATH, "//div[contains(@class, \"re__pr-short-info-item js__pr-config-item\")]")
        date_list = []
        for elem in date_elems:
            date_list.append(elem.text)
        
        publishing_date_list.append(date_list[0])   
        expiry_date_list.append(date_list[1])
        news_category_list.append(date_list[2])

        """ description_elems = driver.find_elements(By.XPATH, "//div[contains(@class, \"re__section-body re__detail-content js__section-body js__pr-description js__tracking\")]")
        for elem in description_elems:
            description_list.append(elem.text)  """
            
class Property:
    def __init__(self):
        self.dien_tich = None
        self.mat_tien = None
        self.muc_gia = None
        self.so_tang = None
        self.so_phong_ngu = None
        self.so_toilet = None
        self.phap_ly = None
        self.noi_that = None
        self.huong_nha = None
        self.huong_ban_cong = None
        self.duong_vao = None

def extract_features_information(data):
    information = Property()
    for item in data:
        lines = item.split(',\t')
        for line in lines:
            key, value = line.split('\n', 1)
            key = key.strip()
            value = value.strip()

            if key == 'Diện tích':
                information.dien_tich = value
            elif key == 'Mặt tiền':
                information.mat_tien = value
            elif key == 'Mức giá':
                information.muc_gia = value
            elif key == 'Số tầng':
                information.so_tang = value
            elif key == 'Số phòng ngủ':
                information.so_phong_ngu = value
            elif key == 'Số toilet':
                information.so_toilet = value
            elif key == 'Pháp lý':
                information.phap_ly = value
            elif key == 'Nội thất':
                information.noi_that = value
            elif key == 'Hướng nhà':
                information.huong_nha = value
            elif key == 'Hướng ban công':
                information.huong_ban_cong = value
            elif key == 'Đường vào':
                information.duong_vao = value

    return information

def extract_date_information(data):
    _, value = data.split('\n', 1)
    return value

def data_processing(feature_list, address_list, publishing_date_list, expiry_date_list, news_category_list):
    # Trích xuất thông tin và tạo danh sách các đối tượng Property
    properties = []
    for temp in feature_list:
        property_info = extract_features_information([temp])
        properties.append(property_info)
        
    publishing_dates = []
    for temp in publishing_date_list:
        publishing_dates_info = extract_date_information(temp)
        publishing_dates.append(publishing_dates_info)

    expiry_dates = []
    for temp in expiry_date_list:
        expiry_dates_info = extract_date_information(temp)
        expiry_dates.append(expiry_dates_info)
        
    news_categories = []
    for temp in news_category_list:
        news_categories_info = extract_date_information(temp)
        news_categories.append(news_categories_info)
        
    # Tạo DataFrame từ thuộc tính của các đối tượng Property
    data = {
        'Diện tích': [prop.dien_tich for prop in properties],
        'Mặt tiền': [prop.mat_tien for prop in properties],
        'Mức giá': [prop.muc_gia for prop in properties],
        'Số tầng': [prop.so_tang for prop in properties],
        'Số phòng ngủ': [prop.so_phong_ngu for prop in properties],
        'Số toilet': [prop.so_toilet for prop in properties],
        'Pháp lý': [prop.phap_ly for prop in properties],
        'Nội thất': [prop.noi_that for prop in properties],
        'Hướng nhà': [prop.huong_nha for prop in properties],
        'Hướng ban công': [prop.huong_ban_cong for prop in properties],
        'Đường vào': [prop.duong_vao for prop in properties],
        'Ngày đăng': [date for date in publishing_dates], 
        'Ngày hết hạn': [date for date in expiry_dates],
        'Loại tin': [category for category in news_categories],
        'Địa chỉ': [address for address in address_list]
        #'Mô tả': [description for description in description_list]
    }

    df = pd.DataFrame(data)
    return df


def get_page(num_pages, page_list):
    i = 0
    count = 0

    while i < num_pages:
        if count == 0:
            options = uc.ChromeOptions()
            options.headless = False
            driver = driver = uc.Chrome(options=options)
        
        url = page_list[i]
        driver.get(url)
        sleep(random.randint(3, 7))
        
        accom_categories = driver.find_elements(By.CLASS_NAME, value='js__product-link-for-product-id')
    
        accoms = []
        for category in accom_categories:
            accoms_url = category.get_attribute('href')
            accoms_title = category.text
            accoms.append({'Title': accoms_title, 'URL': accoms_url}) # accoms -> dictionary
        
        # Danh sách dữ liệu mẫu
        feature_list = []
        address_list = []
        #description_list = []
        publishing_date_list = []
        expiry_date_list = []
        news_category_list = []

        # Lấy dữ liệu từ các trang
        get_information(driver, accoms, feature_list, address_list, publishing_date_list, expiry_date_list, news_category_list)
        
        # Tạo DataFrame và lưu vào tệp CSV
        df = data_processing(feature_list, address_list, publishing_date_list, expiry_date_list, news_category_list)
        
        # Lưu vào file csv         
        df.to_csv('shophousenhaphothuongmai_thuduc_page{}.csv'.format(i+1), index=False, encoding='utf-8-sig')

        count+=1
        
        if count == 11: # Mỗi lần mở chrome chỉ crawl 10 trang
            count = 0
            driver.close()
        
        i+=1
  

num_pages = 1
base_url = "https://batdongsan.com.vn/ban-shophouse-nha-pho-thuong-mai-thu-duc"
page_list = get_page_list(num_pages, base_url)
get_page(num_pages, page_list)


