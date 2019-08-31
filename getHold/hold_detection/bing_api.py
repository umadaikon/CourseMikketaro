# -*- coding: utf-8 -*-
import http.client
import urllib.parse
import requests
import json
import os
import math

# Replace the subscriptionKey string value with your valid subscription key.
subscriptionKey = "12b128451d1e4e59b5783d68e2b59051"

# Verify the endpoint URI.  At this writing, only one endpoint is used for Bing
# search APIs.  In the future, regional endpoints may be available.  If you
# encounter unexpected authorization errors, double-check this value against
# the endpoint for your Bing search instance in your Azure dashboard.
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/images/search"

term = "ホールド　スクリューオン　ボルトオン　ボルダリング"
save_dir_path = "./"
count = 0

num_imgs_required = 1000 # Number of images you want. The number to be divisible by 'num_imgs_per_transaction'
num_imgs_per_transaction = 150 # default 30, Max 150
offset_count = math.floor(num_imgs_required / num_imgs_per_transaction)

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def BingImageSearch(search):
    "Performs a Bing image search and returns the results."
    for offset in range(offset_count):

        params = urllib.parse.urlencode({
            # Request parameters
            'q': term,
            'mkt': 'ja-JP',
            'count': num_imgs_per_transaction,
            'offset': offset * num_imgs_per_transaction # increment offset by 'num_imgs_per_transaction' (for example 0, 150, 300)
        })
	
        headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
        conn = http.client.HTTPSConnection(host)
        query = urllib.parse.quote(search)
        #conn.request("GET", path + "?q=" + query, headers=headers)
        conn.request("GET", path + "?%s" % params, "{body}", headers)
        response = conn.getresponse()
        headers = [k + ": " + v for (k, v) in response.getheaders()
                   if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
        data = response.read()
        conn.close()
    return headers, data.decode("utf8")
	

def make_img_path(save_dir_path, url, term):
    save_img_path = os.path.join(save_dir_path, term)
    make_dir(save_img_path)
    global count
    count += 1

    file_extension = os.path.splitext(url)[-1]
    if file_extension.lower() in ('.jpg', '.jpeg', '.gif', '.png', '.bmp'):
        full_path = os.path.join(save_img_path, str(count)+file_extension)
        return full_path
    else:
        raise ValueError('Not applicable file extension')


def download_image(url, timeout=10):
    response = requests.get(url, allow_redirects=True, timeout=timeout)
    if response.status_code != 200:
        error = Exception("HTTP status: " + response.status_code)
        raise error

    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        error = Exception("Content-Type: " + content_type)
        raise error

    return response.content


def save_image(filename, image):
    with open(filename, "wb") as fout:
        fout.write(image)


def main():
    if len(subscriptionKey) == 32:
        try:
            make_dir(save_dir_path)
            url_list = []

            print('Searching images for: ', term)
            headers, result = BingImageSearch(term)
			
        except Exception as err:
            print("[Errno {0}] {1}".format(err.errno, err.strerror))
        else:
            data = json.loads(result)

            #print(data)
            for values in data['value']:
                    unquoted_url = urllib.parse.unquote(values['contentUrl'])
                    url_list.append(unquoted_url)

        for url in url_list:
            try:
                img_path = make_img_path(save_dir_path, url, term)
                image = download_image(url)
                save_image(img_path, image)
                print('saved image... {}'.format(url))
            except KeyboardInterrupt:
                break
            except Exception as err:
                print("%s" % (err))
    else:
        print("Invalid Bing Search API subscription key!")
        print("Please paste yours into the source code.")


if __name__ == '__main__':
    main()