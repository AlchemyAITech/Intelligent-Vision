from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('http://127.0.0.1:8000')
        # Wait for potential Vue rendering
        page.wait_for_timeout(2000)
        # Click on '项目管理'
        page.evaluate("() => { const links = Array.from(document.querySelectorAll('div, span, a')).filter(el => el.textContent.includes('项目管理')); if(links.length > 0) links[links.length - 1].click(); }")
        page.wait_for_timeout(2000)
        page.screenshot(path='test_ui.png')
        print("Screenshot captured to test_ui.png")
        browser.close()

if __name__ == '__main__':
    run()
