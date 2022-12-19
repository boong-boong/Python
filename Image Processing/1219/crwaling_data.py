from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool
import pandas as pd
import os
import time
import urllib.request

# 키워드 가져오기
keys = pd.read_csv('./keyword.txt', encoding='utf-8', names=['keyword'])
keyword = []
[keyword.append(keys['keyword'][x]) for x in range(len(keys))]

print(keyword)


# 이미지 저장할 폴더 구성
def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error creating folder', dir)


# image download 함수
def image_download(keyword):
    create_folder('./' + keyword + '/')

    # chromedriver 가져오기
    driver = webdriver.Chrome()
    driver.implicitly_wait(3)

    print('keyword: ' + keyword)
    driver.get('https://www.google.com')
    keywords = driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
    keywords.send_keys(keyword)
    # driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[2]/div[2]/div[5]/center/input[1]').click()
    keywords.send_keys(Keys.RETURN)

    # 이미지 탭 클릭
    keywords = driver.find_element(By.XPATH, '//*[@id="hdtb-msb"]/div[1]/div/div[2]/a').click()

    # 스크롤 내리기 -> 결과 더보기 버튼 클릭
    # print("스크롤 ..... ", keyword)
    elem = driver.find_element(By.TAG_NAME, 'body')
    for i in range(100):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.4)

    try:
        # //*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input
        driver.find_element_by_xpath(
            '//*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
        for i in range(100):
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.4)
    except:
        pass

    images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i.Q4LuWd")
    print(keyword + ' 찾은 이미지 개수:', len(images))

    links = []
    for i in range(1, len(images)):
        try:
            # //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
            # //*[@id = "islrg"]/div[1]/div[40]/a[1]/div[1]/img
            driver.find_element(By.XPATH,
                '//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click()
            links.append(driver.find_element(By.XPATH,
                '//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img').get_attribute('src'))
            print(keyword + '링크 수집 중 ... number :  ' +
                  str(i) + '/' + str(len(images)))
        except:
            continue
    fordidden = 0
    for index, i in enumerate(links):
        try:
            url = i
            start = time.time()
            urllib.request.urlretrieve(
                url, "./" + keyword + '/' + str(index-fordidden) + ".jpg")
            print(str(index+1) + "/" + str(len(links)) + ' ' + keyword +
                  '다운로드 중 ..... Download time : ' + str(time.time()-start)[:5] + '초')
        except:
            fordidden += 1
            continue
    print(keyword+'---다운로드 완료---')


if __name__ == '__main__':
    pool = Pool(processes=5)
    pool.map(image_download, keyword)
