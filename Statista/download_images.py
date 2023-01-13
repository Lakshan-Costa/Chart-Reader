import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.support import ui
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import webbrowser
import os
import selenium

def addToClipBoard(text):
    command = 'echo ' + text.strip() + '| clip'
    os.system(command)

driver = webdriver.Firefox()

url = "https://www.statista.com/search/?q=statistics&qKat=newSearchFilter&sortMethod=idrelevance&isRegionPref=1&statistics=1&expert-tools=1&cmo=1&mmo=1&co=1&tmo=1&amo=1&io=1&hmo=1&dmo=1&accuracy=and&isoregion=0&isocountrySearch=&category=0&interval=0&archive=1"
driver.get(url)

wait = ui.WebDriverWait(driver, 10)
#wait.until(lambda driver: driver.find_element_by_id('onetrust-accept-btn-handler'))

time.sleep(5)
pyautogui.click(433, 456, 1, 1, 'left')

#1209 208
#468 432
#405 492
#473 596

pyautogui.click(1209, 208, 1, 1, 'left')
time.sleep(1)
pyautogui.click(468, 432, 1, 1, 'left')
pyautogui.write('dinujaya.20210418@iit.ac.lk')
pyautogui.click(405, 492, 1, 1, 'left')
pyautogui.write('Rubiks cube#123#')
pyautogui.click(473, 596, 1, 1, 'left')

# scroll down
for i in range(20):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

while (True):
    elements = driver.find_elements(By.CLASS_NAME, "list__itemWrap")
    count = len(elements)
    count_now = 0
    # Iterate through the elements
    for element in elements:
        
        link = element.get_attribute("href")
        addToClipBoard(link)
        pyautogui.keyDown('ctrl')  # hold down the shift key
        pyautogui.press('t')     # press the left arrow key
        pyautogui.press('v')     # press the left arrow key
        pyautogui.keyUp('ctrl')    # release the shift key
        pyautogui.press('enter')
        time.sleep(10)
        pyautogui.click(823, 241, 1, 1, 'left')
        time.sleep(1)
        pyautogui.click(1086, 433, 1, 1, 'left')
        pyautogui.click(1086, 420, 1, 1, 'left')
        time.sleep(4)    
        pyautogui.click(527, 27, 1, 1, 'left')
        count_now+=1
        print(count_now , count)
    time.sleep(2)
    pyautogui.click(908, 845, 1, 1, 'left')
  
# Close the driver
driver.quit()
