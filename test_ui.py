from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('http://127.0.0.1:32100/frontend/index.html')
        page.wait_for_timeout(2000)
        
        # Click Project Management
        print("Clicking Project Management...")
        page.evaluate("() => { const links = Array.from(document.querySelectorAll('div, span, a')).filter(el => el.textContent.includes('项目管理')); if(links.length > 0) links[links.length - 1].click(); }")
        page.wait_for_timeout(2000)
        
        # Open Project directly by clicking a card
        print("Opening first project...")
        page.evaluate("() => { const cards = document.querySelectorAll('div[style*=\"cursor: pointer\"]'); if(cards.length > 0) cards[0].click(); }")
        page.wait_for_timeout(2000)
        
        # Get Datasets
        print("Checking Datasets dropdown...")
        ds_options = page.evaluate("() => { const selects = document.querySelectorAll('select'); if(selects.length > 0) return Array.from(selects[0].options).map(o => o.text); return []; }")
        print(f"Datasets found: {ds_options}")
        
        # Select first actual dataset
        if len(ds_options) > 1:
            val = page.evaluate("() => { const sel = document.querySelectorAll('select')[0]; sel.selectedIndex = 1; sel.dispatchEvent(new Event('change')); return sel.value; }")
            print(f"Selected dataset: {val}")
            page.wait_for_timeout(2000)
            
            # Check annotation version dropdown
            print("Checking Annotation Version dropdown...")
            ver_options = page.evaluate("() => { const selects = document.querySelectorAll('select'); if(selects.length > 1) return Array.from(selects[1].options).map(o => o.text); return []; }")
            print(f"Versions found: {ver_options}")
            
        page.screenshot(path='test_ui_final.png')
        browser.close()

if __name__ == '__main__':
    run()
