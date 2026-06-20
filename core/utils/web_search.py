"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import os
import logging

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

import requests
from bs4 import BeautifulSoup
import re

# Reduce noisy dependency logs in normal runs.
logging.getLogger("ddgs.ddgs").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _clean_query(raw_query: str, max_len: int = 220) -> str:
    q = str(raw_query or "").strip()
    if not q:
        return ""

    # Normalize tuple-like chat message artifacts: ('user', '...')
    m = re.match(r"^\(\s*'[^']+'\s*,\s*'(.*)'\s*\)$", q)
    if m:
        q = m.group(1)

    q = q.replace("\n", " ")
    q = re.sub(r"\s+", " ", q).strip()

    # Drop noisy punctuation that hurts web retrieval quality.
    q = q.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    q = re.sub(r"\s+", " ", q).strip()

    if len(q) > max_len:
        q = q[:max_len].rsplit(" ", 1)[0]
    return q


def _ddgs_text_search(query: str, max_results: int) -> list:
    """Run DDGS with backend fallback and return normalized rows."""
    if not DDGS:
        return []

    cleaned = _clean_query(query)
    if not cleaned:
        return []

    # Try supported backends first to improve reliability under network restrictions.
    backends = ["auto", "duckduckgo", "brave", "google", "yandex", "wikipedia", "yahoo", "mojeek", "grokipedia"]
    rows = []
    for backend in backends:
        try:
            try:
                ddgs_client = DDGS(timeout=8)
            except TypeError:
                ddgs_client = DDGS()

            with ddgs_client as ddgs:
                try:
                    data = list(ddgs.text(cleaned, max_results=max_results, backend=backend))
                except TypeError:
                    # Some DDGS versions do not support backend argument.
                    data = list(ddgs.text(cleaned, max_results=max_results))

            if data:
                for r in data[:max_results]:
                    if isinstance(r, dict):
                        rows.append(
                            {
                                "title": str(r.get("title") or ""),
                                "href": str(r.get("href") or r.get("url") or ""),
                                "body": str(r.get("body") or r.get("snippet") or ""),
                            }
                        )
                if rows:
                    return rows
        except Exception:
            continue
    return []

def perform_search(query, max_results=5):
    """Fetch search results using DDGS, Bing, local component data, then ArXiv."""

    query = _clean_query(query)

    # Check for Literature Search intent
    is_literature = any(k in query.lower() for k in ["paper", "literature", "methodology", "review", "state-of-the-art"])
    if is_literature:
        return perform_search_arxiv(query, max_results)

    # Check for Component Search intent
    is_component = any(k in query.lower() for k in ["mosfet", "diode", "capacitor", "price", "datasheet", "stock", "component", "selection"])
    
    # ---------------------------------------------------------
    # STRATEGY 1: DDGS Multi-engine (default ON)
    # ---------------------------------------------------------
    # Set PEMAS_ENABLE_DDGS=0 only when temporary network troubleshooting is needed.
    enable_ddgs = os.environ.get("PEMAS_ENABLE_DDGS", "1") != "0"
    if DDGS and enable_ddgs:
        try:
            print(f"DEBUG: Performing Web Search via DuckDuckGo...", end=" ")
            # Two attempts: raw query then a simplified query with site restrictions removed.
            attempts = [query]
            if "site:" in query:
                attempts.append(re.sub(r"site:[^\s]+", "", query).replace(" OR ", " "))

            for q_try in attempts:
                results = _ddgs_text_search(q_try, max_results=max_results)
                if results:
                    final_res = []
                    for r in results:
                        final_res.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}")
                    print(f"Success ({len(final_res)} results)")
                    return final_res
            print("No DDGS hits after retries.")
        except Exception as e:
            print(f"Failed ({str(e)}). Falling back...")
    elif DDGS and not enable_ddgs:
        print("DEBUG: DDGS search disabled by PEMAS_ENABLE_DDGS=0. Skipping to Bing/local search.")
    
    # ---------------------------------------------------------
    # STRATEGY 2: Bing (CN/Global Backup)
    # ---------------------------------------------------------
    base_url = str(os.getenv("PE_MAS_WEB_SEARCH_URL") or "").strip()
    if not base_url:
        return search_local_component_db(query) if is_component else perform_search_arxiv(query, max_results)
    print("DEBUG: Performing Web Search via configured backend...", end=" ")
    
    headers = {
        "User-Agent": os.getenv("PE_MAS_WEB_SEARCH_USER_AGENT", "PE-MAS/1.0"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    
    # Optimize query for Bing only if we didn't do it globally yet (passed in query is raw usually)
    # But usually the caller modifies it? No, query here is passed arg.
    # The previous code modified 'query' locally. Restoring that optimization:
    search_query = query
    if is_component and "site:" not in search_query:
        search_query += " (site:digikey.com OR site:mouser.com OR site:lcsc.com OR site:infineon.com)"

    params = {'q': search_query}
    
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Verify=False to bypass SSL errors (common in corporate/CN networks)
        resp = requests.get(base_url, params=params, headers=headers, timeout=5, verify=False)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Bing search results usually in <li class="b_algo">
            results_items = soup.find_all('li', class_='b_algo')
            
            final_res = []
            for item in results_items[:max_results]:
                 h2 = item.find('h2')
                 if h2 and h2.find('a'):
                     title = h2.get_text()
                     link = h2.find('a')['href']
                     try:
                        desc = item.find('div', class_='b_caption').find('p').get_text()
                     except:
                        desc = "No description."
                     final_res.append(f"Title: {title}\nLink: {link}\nSnippet: {desc}")
            
            if final_res:
                print(f"Success ({len(final_res)} results)")
                return final_res
            else:
                print("No results found.")
            
    except Exception as e:
        print(f"Error ({str(e)})")

    # ---------------------------------------------------------
    # STRATEGY 3: Local Knowledge Base (SOTA Offline Fallback)
    # ---------------------------------------------------------
    if is_component:
        print("DEBUG: Active Internet Search Failed. Switching to Local Component Database.")
        try:
             res = search_local_component_db(query)
             return res
        except Exception as e:
            print(f"Local DB Search Error: {e}")
            # Fallback to plausible mock if even local DB fails
            return [
                "Title: LCSC Electronics - Global Electronic Component Distributor\nLink: https://www.lcsc.com\nSnippet: Huge inventory of MOSFETs, Diodes, and Controllers. Direct pricing and datasheet availability.",
                "Title: DigiKey Electronics - Electronic Components Distributor\nLink: https://www.digikey.com\nSnippet: World's largest selection of electronic components. Datasheets for IP65R (CoolMOS) and SiC devices available."
            ]
    
    # Generic fallback
    return perform_search_arxiv(query, max_results)

def search_local_component_db(query):
    """
    intelligent local search using the embedded pandas database.
    Parses query for V, I, and Type constraints.
    """
    import pandas as pd
    import re
    
    # Locate DB
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # core/utils/ -> core/knowledge/component_db/
    db_path = os.path.join(current_dir, "../knowledge/component_db/single_fets__mosfets.csv")
    
    if not os.path.exists(db_path):
        return ["Error: Local Component Database not found."]
        
    df = pd.read_csv(db_path)
    
    # Extract Voltage (e.g. 650V, 600 V)
    v_req = 0
    v_match = re.search(r'(\d+)\s*V', query, re.IGNORECASE)
    if v_match:
        v_req = float(v_match.group(1))
        
    # Extract Current (e.g. 10A, 5 A)
    i_req = 0
    i_match = re.search(r'(\d+(\.\d+)?)\s*A', query, re.IGNORECASE)
    if i_match:
        i_req = float(i_match.group(1))
        
    # Filter
    results = []
    
    # Helper to parse '100 V' or '9.8A (Ta), 37A (Tc)' to max value
    def parse_val(val_str):
        if not isinstance(val_str, str): return 0
        # Find all float-like numbers
        matches = re.findall(r'(\d+(?:\.\d+)?)', val_str)
        if not matches:
            return 0
        # Convert to floats and take max (e.g., for "9A, 37A" take 37)
        vals = [float(m) for m in matches]
        return max(vals)

    # Apply Filters
    matches = df.copy()
    
    if v_req > 0:
        # Filter: V_ds >= V_req * 0.9
        matches['V_parsed'] = matches["Drain to Source Voltage (Vdss)"].apply(parse_val)
        matches = matches[matches['V_parsed'] >= v_req * 0.9]
        
    if i_req > 0:
         # Filter: Id >= I_req * 0.8 (Allow 20% margin for Ta vs Tc confusion)
         matches['I_parsed'] = matches["Current - Continuous Drain (Id) @ 25°C"].apply(parse_val)
         matches = matches[matches['I_parsed'] >= i_req * 0.8]
         
    # Logic for Component Type (N-Channel vs P-Channel)
    if "n-channel" in query.lower() or "n-ch" in query.lower():
         matches = matches[matches['FET Type'].str.contains('N-Channel', case=False, na=False)]
    elif "p-channel" in query.lower() or "p-ch" in query.lower():
         matches = matches[matches['FET Type'].str.contains('P-Channel', case=False, na=False)]

    # Sort by price. If missing prices, treat as 0 or infinity?
    # Better to prioritize Available Stock first, then Price.
    # Parse Stock "60,221" -> 60221
    def parse_stock(val):
        if not isinstance(val, str): return 0
        return float(re.sub(r'[^\d]', '', val))
        
    matches['Stock_val'] = matches['Stock'].apply(parse_stock)
    matches = matches[matches['Stock_val'] > 0] # Must be in stock
    
    matches['Price_parsed'] = matches['Price'].apply(parse_val)
    matches = matches.sort_values('Price_parsed', ascending=True)

    # Format top 3 results
    top_n = matches.head(3)
    
    for _, row in top_n.iterrows():
        title = f"{row['Mfr Part #']} - {row['Description']}"
        link = row['Datasheet']
        snippet = (f"Manufacturer: {row['Mfr']} | "
                   f"Vdss: {row['Drain to Source Voltage (Vdss)']} | "
                   f"Id: {row['Current - Continuous Drain (Id) @ 25°C']} | "
                   f"RdsOn: {row['Rds On (Max) @ Id, Vgs']} | "
                   f"Price: ${row['Price']} | "
                   f"Stock: {row['Stock']}")
        results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")
        
    if not results:
        # Relax constraints? 
        # If specific type (N/P) filtered everything, try ignore type? No, dangerous.
        # Just return generic.
        return ["Title: Generic High Power MOSFET\nLink: https://www.digikey.com\nSnippet: No exact local match found for specific voltage/current, but generic alternatives available."]
    
    return results


def perform_search_arxiv(query, max_results=5):
    """
    Search ArXiv for scientific papers (Robust Literature Search).
    """
    print(f"DEBUG: Performing Literature Search via ArXiv API...")
    import xml.etree.ElementTree as ET
    try:
        import requests 
    except ImportError:
        # Fallback if requests is missing (unlikely but safe)
        requests = None
    
    # Generic fallback response - used immediately if no requests or network fails
    fallback_resp = [
        "[BOOK] Fundamentals of Power Electronics (Erickson & Maksimovic)\nLink: https://link.springer.com/book/10.1007/b100747\nAbstract: Standard reference for converter design including Flyback CCM/DCM analysis."
    ]

    # Quick check for internet (optional, but good practice)
    # But let's just try-except the request.

    if not requests:
        return fallback_resp
    
    base_url = str(os.getenv("PE_MAS_ARXIV_API_URL") or "").strip()
    if not base_url:
        return fallback_resp
    # Clean query for ArXiv (remove site:...)
    clean_query = query.split('site:')[0].strip()
    # Remove parens
    clean_query = clean_query.replace('(', '').replace(')', '').replace('OR', '')
    
    params = {
        "search_query": f"all:{clean_query}",
        "start": 0,
        "max_results": max_results
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5) # Short timeout
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        entries = []
        # Atom Namespace is critical for parsing
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            link = entry.find('atom:id', ns).text.strip()
            published = entry.find('atom:published', ns).text.strip()[:10]
            
            entries.append(f"[PAPER] {title} ({published})\nLink: {link}\nAbstract: {summary[:200]}...")
            
        if not entries:
             # If API returns empty but no error
             return ["[INFO] No specific papers found on ArXiv for this query."]
            
        return entries
        
    except Exception as e:
        print(f"ArXiv Search Failed: {str(e)}")
        # Graceful fallback so agent doesn't crash or return "Rubbish" empty lists
        # We return a generic 'Textbook' reference which is always valid context
        return fallback_resp


