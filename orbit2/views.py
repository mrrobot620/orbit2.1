from django.shortcuts import render , HttpResponse , redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate , login , logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.db.models import Count
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect
from django.utils import timezone
from django.core.serializers.json import DjangoJSONEncoder
import json
from base64 import b64decode 
import csv
import os
from io import BytesIO
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponse
import uuid
from django.core.files.storage import default_storage
from django.conf import settings
import os
from io import BytesIO
from django.db.models import Q
from django.utils.timezone import localtime, make_aware
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pytesseract
import random
from selenium import webdriver
import os 
from select import select
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC3
from selenium.webdriver.common.action_chains import ActionChains
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import tensorflow as tf
import os
from idna import valid_contextj
from datetime import datetime, timedelta
import logging
import shutil
import requests
from requests_html import HTMLSession
import threading
import re
from .models import Pendency , SearchHistory , ReconciliationRecord
import cv2
import io

from webdriver_manager.chrome import ChromeDriverManager


tf.config.threading.set_inter_op_parallelism_threads(10)

op = webdriver.ChromeOptions()
# op.add_argument('--headless=new')
prefs = {
    'profile.default_content_settings.popups': 0,
    'download.default_directory' : r"/home/administrator/cbs_bag_hold/data",
    'directory_upgrade': True
}
op.add_experimental_option('prefs' , prefs)
driver = webdriver.Chrome(executable_path=r"C:\Users\abhishek.h1.FKIPL\Downloads\orbit-Prod\chromedriver.exe", options=op)

inception_model = InceptionV3(weights='imagenet')
model = VGG16(weights='imagenet', include_top=False)

session = HTMLSession()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)



def extract_features(query_image_path):
    try:
        img = Image.open(query_image_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        features = model.predict(img_array)
        features = np.reshape(features, (7*7*512,))
        decoded_features = features.tobytes()
        decoded_features = np.frombuffer(decoded_features)
        return decoded_features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    sim1 = dot / norm
    sim  = round((sim1 + 1) / 2 * 100 )
    return sim

def find_similar_images(query_img_path, target_img_path, top_k=5):
    query_features = extract_features(query_img_path)
    target_features = np.frombuffer(target_img_path)

    if query_features is None or target_features is None:
        return None
    
    print("Query features shape:", query_features.shape)
    print("Target features shape:", target_features.shape)

    similarity = cosine_similarity(query_features, target_features)
    return similarity

def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request , username=username , password=password)
        if user is not None:
            login(request , user)
            return redirect('home')
        else:
            return render(request , 'login.html' , {'error': "Invalid Username or Password"})
    return render(request , 'login.html')


def home(request):
    return render(request , 'home.html')
    
def add_orphan_page(request):
    return render(request , 'add_orphan.html')

def add_pendency(request):
    tids = [] 
    if request.method == "POST":
        tids_input = request.POST.get("tracking_ids", "")
        tids = tids_input.split() 
        print(tids)
        threads = []
        for id in tids:
            thread = threading.Thread(target=download_tracking_page, args=(id,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    return render(request, "pendency.html", {'tids': tids})


def is_white(color, threshold=200):
    return all(value >= threshold for value in color)

def extract_dominant_color(img_array, k=3):
    img_array_flat = img_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_array_flat)

    dominant_color_centers = kmeans.cluster_centers_.astype(int)

    dominant_colors = [tuple(color) for color in dominant_color_centers if not is_white(color)]

    if not dominant_colors:
        dominant_colors = [tuple(color) for color in dominant_color_centers[1:]]

    return dominant_colors

@csrf_exempt
def classify_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        img = Image.open(uploaded_image)
        img = img.resize((299, 299))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = preprocess_input(img_array)
        predictions = inception_model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        classification_result = decoded_predictions[0][1]
        dominant_colors = extract_dominant_color(np.array(img))
        text = pytesseract.image_to_string(img)
        print(f"Text:  {text}")
        dominant_colors = [[int(value) for value in color] for color in dominant_colors]
        result = {
            'classification_result': classification_result,
            'dominant_colors': dominant_colors,
            "extracted_text": text,
        }
        return JsonResponse(result, encoder=DjangoJSONEncoder)
    return JsonResponse({'error': 'Invalid request'})


def index(request):
    return render(request, 'add_orphan.html')

def profile(request):
    return render(request , 'profile.html')

def logout_view(request):
    logout(request)
    return redirect('login_view')


def login_flo():
    driver.get("http://10.24.2.16/fklshipping/")
    time.sleep(5)
    username = driver.find_element(By.XPATH , "/html/body/div[2]/div[2]/div/div/form/div/div[4]/input[1]")
    username.send_keys("ca.3006878")

    password = driver.find_element(By.XPATH , "/html/body/div[2]/div[2]/div/div/form/div/div[4]/input[2]")
    password.send_keys("4Xss9kiMMy")
    time.sleep(2)
    try:
        cross = driver.find_element(By.XPATH , "/html/body/div[4]/div/button")
        cross.click()
    except:
        print("Cross Button Failed")
    time.sleep(1)
    submit = driver.find_element(By.XPATH , "/html/body/div[2]/div[2]/div/div/form/div/div[4]/div[4]/button/span")
    submit.click()
    time.sleep(10)

def select_facility():
    facility_dropdown = driver.find_element(By.XPATH , "/html/body/div[3]/div/div[2]/div[1]/form/div[1]/div/a")
    facility_dropdown.click()
    facility_dropdown.send_keys("YKB")
    facility_dropdown.send_keys(Keys.RETURN)
    facility_submit = driver.find_element(By.XPATH , "/html/body/div[3]/div/div[2]/div[1]/form/div[3]/input")
    facility_submit.click()
    tracking = driver.find_element(By.XPATH , "/html/body/div[1]/div[3]/div/ol/li[3]/a/span").click()
    time.sleep(3)


def extract_images_info(data):
    images_info = []
    if 'data' in data and 'staticContentInfo' in data['data']:
        static_content_info = data['data']['staticContentInfo']
        for content in static_content_info.get('staticcontents', []):
            for trans_content in content.get('transContents', []):
                attribute_values = trans_content.get('attributeValues', {})
                path_values = attribute_values.get('path', {}).get('valuesList', [])
                if path_values:
                    image_url = path_values[0].get('value', '')
                    if image_url.startswith("http:/"):
                        if image_url[6:].startswith("//"):
                            image_url = "http:" + image_url[6:]
                        else:
                            image_url = "http://" + image_url[6:]
                    images_info.append({'image_url': image_url})
    return images_info

def download_image(image_url, filename):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        with open(image_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Image downloaded successfully: {image_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code} Filename: {filename}")
    

def download_tracking_page(id):
    uploaded_ids = []
    regex_pattern = r'\b[A-Z0-9]{16}\b'
    response = session.get(f"http://10.24.2.16/fklshipping/shipments/track/?id={id}&ts=1705207057878")
    a = response.html.html
    if a is not None:
        ids = re.findall(regex_pattern, a)
        pid = ids[-1]
        print(pid)
        try:
            api = requests.get(f"http://10.83.47.208/v2/product/xif0q/{pid}")
            raw_api = api.json()
            images_link = extract_images_info(raw_api)
            itm_id = raw_api['data']['relationships']['ITEMIZATION']['DEFAULT'][0]['archaicId']
            related_pids = raw_api['data']['relationships']["ITEMIZATION"]['DEFAULT'][0]['relatedEntities']
            metadata = raw_api.get("metadata", {})
            product_attributes1 = metadata.get("productAttributes", {})
            vertical = product_attributes1.get("vertical", "")
            data = raw_api.get('data', {})
            product_attributes = data.get('productAttributes', {})
            attribute_map = product_attributes.get('attributeMap', {})
            brand_map = attribute_map.get("brand", {})
            brand_name = brand_map.get("value", "")
            keys_to_extract = ['description', 'color', 'weight', 'height', 'brand', 'keywords', 'price']
            result_data = {}
            for key in keys_to_extract:
                value = attribute_map.get(key)
                result_data[key] = value
            if id not in uploaded_ids:
                try:
                    existing_product = Pendency.objects.get(tid=id)
                    print(f"Product with Tid {id} already exists in the database, skipping...")
                except Pendency.DoesNotExist:
                    try:
                        if images_link:
                            image_url = images_link[0].get('image_url')
                            download_image(image_url, f"{id}.jpg")
                            print(f"Image Downloaded: {id}")
                        else:
                            print(f"No images found for Tid {id}")
                        product = Pendency.objects.create(
                            description=result_data.get('description', ''),
                            color=result_data.get('color', ''),
                            brand=brand_name,
                            keywords=result_data.get('keywords', ''),
                            price=result_data.get('price', 0.0),
                            pid=pid,
                            tid=id,
                            itm_id=itm_id,
                            vertical=vertical,
                            image=f"{id}.jpg"
                        )
                        uploaded_ids.append(id)
                        for related_pid in related_pids:
                            if related_pid not in uploaded_ids:
                                related_product, created = Pendency.objects.get_or_create(pid=related_pid)
                                if created:
                                    print(f"Related product with PID {related_pid} created.")
                                else:
                                    print(f"Related product with PID {related_pid} already exists.")
                                product.related_pids.add(related_product)
                    except Exception as e:
                        print(f"Error saving product to the database: {e}")
            else:
                print(f"ID {id} already uploaded, skipping...")
        except Exception as e:
            print(f"Error in running API: {e}")
    else:
        print(f"Unable to find Pid for Tid {id}")

def open_new_tab():
    driver.execute_script("window.open('', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])

def search_view(request):
    results = []
    unique_verticals = Pendency.objects.values('vertical').distinct()


    if request.method == 'POST':
        lens_pids = extract_pid()
        print(lens_pids)
        brand = request.POST.get('brand')
        uploaded_image = request.FILES.get('image')
        vertical = request.POST.getlist("vertical")
        keywords = request.POST.getlist("keywords")

        query_keywords = keywords[0].split(",")

        if not (uploaded_image and (brand or vertical)):
            print(f"Vertical: {vertical}, Brand: {brand} , Keywords:  {query_keywords} , image: {uploaded_image}")
            return render(request, 'search.html', {'results': results, 'unique_verticals': unique_verticals, 'error_message': 'Invalid input'})

        queryset = Pendency.objects.all()
        # added this to fast forward the search , if pids matches from the result of google search
        if lens_pids is not None:
            lens_pids_set = set(lens_pids)
            queryset = Pendency.objects.annotate(
                matched_pid_itm=Q(pid__in=lens_pids_set) | Q(itm_id__in=lens_pids_set)
             )
            
            queryset = queryset.filter(related_pids__pid__in=lens_pids_set)

            matching_pids = set(queryset.values_list('tid', flat=True))
            print(f"matching pids: ==  {matching_pids}")
            if matching_pids:
                if uploaded_image:
                    uploaded_image_path, unique_filename = handle_uploaded_image(uploaded_image)
                    queryset = queryset.filter(pid__in=matching_pids)
                    print(f"Queryset == {queryset}")
                    results = [{'tid': pendency.tid, 'pid': pendency.pid , 'pendency_image_name':pendency.image.name, 'similarity': "100", 'uploaded_image_name': unique_filename} for pendency in queryset]
                return render(request, 'results.html', {'results': results, 'unique_verticals': unique_verticals})
            else:
                print("Lens Pid Failed")
                pass

        if vertical:
            queryset = queryset.filter(vertical__in=vertical)
            print(f"After Selecting Vertica:  {queryset}")
        if brand:
            queryset = queryset.filter(brand=brand)
        
        keyword_queryset = keyword_query(query_keywords)

        if keyword_query is not None:
            queryset = queryset.filter(pk__in=keyword_queryset.values_list('pk', flat=True))
            
        print(f"Vertical: {vertical}, Brand: {brand} , Keywords:  {query_keywords}")

        unique_brands = Pendency.objects.filter(vertical=vertical).values('brand').distinct()

        if uploaded_image:
            uploaded_image_path, unique_filename = handle_uploaded_image(uploaded_image)
            search_uuid = uuid.uuid4()

            if request.user.is_authenticated:
                user = request.user
                search_parameters = {
                    'brand': brand,
                    'vertical': vertical,
                    'keywords': query_keywords,
                    }
                
                search_history_entry = SearchHistory.objects.create(user=user, reconciled=False, query=search_parameters , filename=unique_filename , uuid =search_uuid)

            print(f"Uploaded Image Path: {uploaded_image_path}")

            if not os.path.exists(uploaded_image_path):
                return render(request, 'search.html', {'results': results, 'unique_verticals': unique_verticals, 'unique_brands': unique_brands, 'error_message': 'Uploaded Image File Does Not Exist!'})
            
            try:
                threads = []
                for pendency in queryset:
                    thread = threading.Thread(target=process_image, args=(uploaded_image_path, pendency, results, unique_filename, search_uuid))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()

                results.sort(key=lambda x: x['similarity'], reverse=True)
                return render(request, 'results.html', {'results': results, 'unique_verticals': unique_verticals, 'unique_brands': unique_brands ,  "uuid":search_uuid})

            except Exception as e:
                print(f"Error processing image: {e}")
                return render(request, 'search.html', {'results': results, 'unique_verticals': unique_verticals, 'unique_brands': unique_brands, 'error_message': 'Error processing image'})

    return render(request, 'search.html', {'results': results, 'unique_verticals': unique_verticals, 'error_message': 'No Results Found!'})

def process_image(uploaded_image_path, pendency, results, unique_filename, search_uuid):
    pendency_image_path = pendency.features
    try:
        existing_pendency = Pendency.objects.get(features=pendency_image_path)
    except Pendency.DoesNotExist:
        print(f"Skipping pendency {pendency.tid} due to missing image file.")
        return

    similarity = find_similar_images(uploaded_image_path, pendency_image_path)
    print(f"Similarity for TID {pendency.tid}: {similarity}")
    results.append({'tid': pendency.tid, 'pid':pendency.pid, 'similarity': similarity, 'uploaded_image_name': unique_filename, 'pendency_image_name': pendency.image.name ,  "uuid":search_uuid})


def keyword_query(keywords):
    q_objects = [Q(keywords__icontains=keyword) for keyword in keywords]

    query = Q()
    for q_obj in q_objects:
        print(f"Tid {q_obj}")
        query |= q_obj

    return Pendency.objects.filter(query)

def handle_uploaded_image(uploaded_image):
    unique_filename = f"{uuid.uuid4().hex[:10]}.jpg"
    uploaded_image_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_images', unique_filename)

    try:
        os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)

        with open(uploaded_image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        return uploaded_image_path , unique_filename

    except Exception as e:
        print(f"Error saving image: {e}")
        return None
    

def get_brands_for_vertical(request):
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == 'GET':
        vertical_param = request.GET.get('vertical', None)

        if vertical_param:
            vertical_list = vertical_param.split(',')  # Convert the comma-separated string to a list
            unique_brands = Pendency.objects.filter(vertical__in=vertical_list).values('brand').distinct()
            brands_list = list(unique_brands.values_list('brand', flat=True))
            print(brands_list)
            return JsonResponse({'brands': brands_list})

    return JsonResponse({'error': 'Invalid request'})

def results_view(request):
    return render(request, 'results.html')

def db_view(request):
    return render(request , 'database.html')

def get_details(request, tid):
    pendency = get_object_or_404(Pendency, tid=tid)
    details_data = {
        'tid': pendency.tid,
        'description': pendency.description,
        'color': pendency.color,
        'brand': pendency.brand,
        'vertical': pendency.vertical,
        'image_url': os.path.join(settings.MEDIA_URL, pendency.image.name),
        "pid": pendency.pid
    }
    return JsonResponse(details_data, encoder=DjangoJSONEncoder)


def download_pendencies(request):
    queryset = Pendency.objects.all()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="pendencies.csv"'
    writer = csv.writer(response)
    writer.writerow(['Vertical', 'Brand', 'PID', 'TID' , 'color',  "Keywords"])
    for pendency in queryset:
        writer.writerow([pendency.vertical, pendency.brand, pendency.pid, pendency.tid , pendency.color ,  pendency.keywords])
    return response

def oid_generator():
    linux_time_now = int(time.time())
    oid = f"YKB_COS_{linux_time_now}"
    return oid


def reconcile_search_history(request):
    if request.method == 'POST':
        uuid = request.POST.get('uuid')

        search_history = get_object_or_404(SearchHistory, user=request.user, uuid=uuid, reconciled=False)
        search_history.reconciled = True
        search_history.save()

        tid = request.POST.get('tid')
        pendency_instance = get_object_or_404(Pendency, tid=tid)
        pendency_instance.delete()

        reconciliation_record = ReconciliationRecord(user=request.user, tid=tid)
        reconciliation_record.save()

        return redirect('search')
    return HttpResponse("Invalid request.")

@csrf_exempt
def upload_image_to_aws(request):
    try:
        uploaded_image = request.FILES['image']
        img = Image.open(uploaded_image)
        if img.format == 'PNG':
            converted_image_io = io.BytesIO()
            img = img.convert('RGB')
            max_size = (800, 600)
            img.thumbnail(max_size)
            img.save(converted_image_io, format='JPEG', quality=70)
            converted_image_io.seek(0)
            image_data = converted_image_io
        else:
            max_size = (800, 600)
            img.thumbnail(max_size)
            image_data = io.BytesIO()
            img.save(image_data, format='JPEG', quality=70)
            image_data.seek(0)
        
        files = {'image': ('a.jpg', image_data)}
    
        upload_url = 'http://65.2.153.92:8000/upload_image/'

        response = requests.post(upload_url, files=files)
        
        if response.status_code == 200:
            google_pids = extract_pid()
            return HttpResponse(f"Image uploaded successfully!  {str(google_pids)}" )
        else:
            return HttpResponse(f"Failed to upload image. Status code: {response.status_code}", status=500)
    except KeyError:
        return HttpResponse("No image file found in the request.", status=400)
    except Exception as e:
        return HttpResponse(f"Error uploading image: {str(e)}", status=500)


def extract_pid():
    start_time = time.time()
    url = "https://lens.google.com/uploadbyurl?url=http://65.2.153.92:8000/images/a.jpg/"
    driver.get(url)
    links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, "a")]
    filtered_links = [link for link in links if link.startswith("https://www.flipkart.com/")]
    pattern = r'pid=([A-Za-z0-9]+)&'
    pattern2 = r'itm([A-Za-z0-9]+)'
    pid_matches = []
    while not pid_matches and time.time() - start_time < 20:
        pid_matches = []
        for link in filtered_links:
            pid_matches.extend(re.findall(pattern, link))
            pid_matches.extend(["ITM" + match.upper() for match in re.findall(pattern2, link)])
    if not pid_matches:
        print("PID extraction timed out.")
        return None
    else:
        print(pid_matches)
        return pid_matches


login_flo()
select_facility()
session_cookie = driver.get_cookies()
print(session_cookie)
selenium_user_agent = driver.execute_script("return navigator.userAgent;")
print(selenium_user_agent)
session.headers.update({"user-agent": selenium_user_agent})
for cookie in driver.get_cookies():
    session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])