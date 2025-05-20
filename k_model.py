import os
import json
import asyncio
import aiohttp
import aiofiles
import logging
import sys
import re
import math
import pandas as pd
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Union
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from unidecode import unidecode
from pybaseball import playerid_lookup
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import numpy as np

# Load CSV data for static projections
batter_df = pd.read_csv("batter_putaway.csv", skiprows=4)
pitcher_putaway_df = pd.read_csv("pitcher_ID_putaway.csv", skiprows=2)

# Clean pitcher data
pitcher_df = pitcher_putaway_df[['Name', 'MLBAM_ID', 'Putaway']]
pitcher_df.columns = ['Name', 'MLBAM_ID', 'Putaway_Pitch']
pitcher_df['Name'] = pitcher_df['Name'].str.strip()
pitcher_putaway_map = dict(zip(pitcher_df['Name'], pitcher_df['Putaway_Pitch']))

batter_df = batter_df[['Name', 'MLBAM_ID', 'K%', 'FB Whiff%', 'FB Putaway%',
                       'Breaking Whiff%', 'Breaking Putaway%',
                       'Offspeed Whiff%', 'Offspeed Putaway%']]
batter_df['Name'] = batter_df['Name'].str.strip()
for col in batter_df.columns[2:]:
    batter_df[col] = pd.to_numeric(batter_df[col], errors='coerce')

# Pitch type to category mapping
PITCH_CATEGORY_MAP = {
    'FF': 'FB', 'FT': 'FB', 'SI': 'FB', 'FC': 'FB', 'FA': 'FB',
    'SL': 'Breaking', 'CU': 'Breaking', 'KC': 'Breaking', 'SK': 'Breaking',
    'CH': 'Offspeed', 'FS': 'Offspeed', 'SC': 'Offspeed'
}



# Load env vars
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("projection.log", encoding='utf-8')
    ]
)

# Constants
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
RATE_LIMITER = asyncio.Semaphore(5)
MAX_FAILURES = 5
SCRAPE_FAILURES = {}

LEAGUE_AVG = {'k_pct': 0.22, 'whiff_pct': 0.28, 'xwoba': 0.320}
LEAGUE_AVG_K = 6.5

TEAM_IDS = {
    "Diamondbacks": 109, "Braves": 144, "Orioles": 110, "Red Sox": 111,
    "Cubs": 112, "White Sox": 145, "Reds": 113, "Guardians": 114,
    "Rockies": 115, "Tigers": 116, "Astros": 117, "Royals": 118,
    "Angels": 108, "Dodgers": 119, "Marlins": 146, "Brewers": 158,
    "Twins": 142, "Yankees": 147, "Mets": 121, "Athletics": 133,
    "Phillies": 143, "Pirates": 134, "Padres": 135, "Giants": 137,
    "Mariners": 136, "Cardinals": 138, "Rays": 139, "Rangers": 140,
    "Blue Jays": 141, "Nationals": 120
}

# Vegas manual fallback lines from your data
class VegasLineFetcher:
    def __init__(self):
        self.manual_lines = {
            "zac gallen": 7.0, "pablo lópez": 6.5,
            "max scherzer": 6.5, "tyler glasnow": 6.5,
            "freddy peralta": 6.0, "nick lodolo": 6.0,
            "mitch keller": 6.5,
        }
        self.cache_expiry = timedelta(hours=6)

    def _get_cache_path(self, pitcher_name: str) -> str:
        safe_name = "".join(c if c.isalnum() else "_" for c in pitcher_name.lower())
        return os.path.join(CACHE_DIR, f"{safe_name}.json")

    async def get_vegas_line(self, pitcher_name: str) -> Tuple[float, str]:
        # Cache disabled for brevity — add if you want
        line = self.manual_lines.get(pitcher_name.lower(), 6.5)
        return line, 'manual-fallback'

vegas_fetcher = VegasLineFetcher()

# --- Helpers ---

def normalize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]", "", unidecode(name.lower()))
    return hashlib.sha256(clean.encode()).hexdigest()

async def save_cache(name: str, data: dict):
    path = os.path.join(CACHE_DIR, f"{normalize_name(name)}.json")
    async with aiofiles.open(path, "w") as f:
        await f.write(json.dumps(data))

async def load_cache(name: str) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"{normalize_name(name)}.json")
    if not os.path.exists(path):
        return None
    async with aiofiles.open(path) as f:
        data = json.loads(await f.read())
    if data.get('expires', 0) > datetime.now().timestamp():
        return data
    return None

def create_default_batter_stats(name: str) -> Dict:
    return {
        'name': name,
        'k_pct': LEAGUE_AVG['k_pct'],
        'whiff_pct': LEAGUE_AVG['whiff_pct'],
        'xwoba': LEAGUE_AVG['xwoba'],
        'hand': 'R',
        'fallback': True
    }

# --- Batter stat scraping ---

async def get_fangraphs_stats(session: aiohttp.ClientSession, name: str) -> Optional[Dict]:
    try:
        url = f"https://www.fangraphs.com/players/{name.lower().replace(' ', '-')}/stats"
        async with session.get(url, headers={"User-Agent": USER_AGENT}) as res:
            if res.status != 200:
                return None
            html = await res.text()
            return await parse_fangraphs_stats(html)
    except Exception as e:
        logging.DEBUG(f"FanGraphs fail for {name}: {e}")
        return None

async def parse_fangraphs_stats(html: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "SeasonStats1"})
    if not table:
        return {}

    rows = table.tbody.find_all("tr")
    mlb_row = next((row for row in rows if row.find("td", {"data-stat": "Level"}).text.strip() == "MLB"), None)
    if not mlb_row:
        return {}

    return {
        "k_pct": float(mlb_row.find("td", {"data-stat": "k_percent"}).text.strip('%')) / 100,
        "whiff_pct": float(mlb_row.find("td", {"data-stat": "swstr_percent"}).text.strip('%')) / 100,
        "xwoba": float(mlb_row.find("td", {"data-stat": "xwOBA"}).text),
        "hand": mlb_row.find("td", {"data-stat": "bats_throws"}).text[0].upper(),
        "pa": int(mlb_row.find("td", {"data-stat": "pa"}).text)
    }

async def get_savant_stats(session: aiohttp.ClientSession, player_name: str) -> Tuple[float, float]:
    try:
        async with RATE_LIMITER:
            search_url = f"https://baseballsavant.mlb.com/search?searchTerm={player_name}"
            async with session.get(search_url, timeout=10, headers={"User-Agent": USER_AGENT}) as res:
                if res.status != 200:
                    return 0.28, 0.320
                search_html = await res.text()

        search_soup = BeautifulSoup(search_html, "html.parser")
        href = None
        for a in search_soup.select("a"):
            if player_name.lower() in a.text.lower():
                href = a['href']
                break
        if not href:
            return 0.28, 0.320

        profile_url = f"https://baseballsavant.mlb.com{href}"
        async with RATE_LIMITER, session.get(profile_url, timeout=10) as res:
            if res.status != 200:
                return 0.28, 0.320
            profile_html = await res.text()

        whiff_pct, xwoba = 0.28, 0.320
        soup = BeautifulSoup(profile_html, 'html.parser')
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                label = cells[0].text.strip()
                value = cells[1].text.strip().replace("%", "")
                if "Whiff%" in label:
                    try:
                        whiff_pct = float(value)/100
                    except:
                        pass
                elif "xwOBA" in label:
                    try:
                        xwoba = float(value)
                    except:
                        pass
        return whiff_pct, xwoba
    except Exception as e:
        logging.error(f"Savant scrape failed for {player_name}: {e}")
        return 0.28, 0.320

async def get_batter_stats(session: aiohttp.ClientSession, name: str) -> Dict:
    if SCRAPE_FAILURES.get(name, 0) > MAX_FAILURES:
        logging.warning(f"Circuit breaker activated for {name}")
        return create_default_batter_stats(name)

    cached = await load_cache(name)
    if cached:
        cached['name'] = name
        return cached

    async with RATE_LIMITER:
        fg_stats = await get_fangraphs_stats(session, name)
        if fg_stats:
            fg_stats['name'] = name
            await save_cache(name, fg_stats)
            return fg_stats

        whiff, xwoba = await get_savant_stats(session, name)
        stats = {
            'name': name,
            'k_pct': LEAGUE_AVG['k_pct'],
            'whiff_pct': whiff,
            'xwoba': xwoba,
            'hand': 'R',
            'pa': 200,
            'fallback': True
        }
        await save_cache(name, stats)
        return stats

# --- Lineup fetching & validation ---

def validate_lineup(lineup: List[str]) -> bool:
    valid_chars = re.compile(r'^[\w\s.\'-]+$')
    unique_names = len(set(name.lower() for name in lineup))
    return (
        isinstance(lineup, list)
        and 8 <= len(lineup) <= 10
        and unique_names == len(lineup)
        and all(
            isinstance(name, str)
            and 3 <= len(name) <= 40
            and len(name.split()) >= 2
            and valid_chars.match(name)
            and not name.lower().endswith("pitcher")
            for name in lineup
        )
    )

async def get_mlb_lineup(session: aiohttp.ClientSession, team_id: int) -> Optional[List[str]]:
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}&date={today}"
        async with session.get(url, timeout=10) as res:
            data = await res.json()

        if not data.get('dates'):
            return None

        games = data['dates'][0]['games']
        if not games:
            return None

        game = games[0]
        status = game.get('status', {}).get('statusCode', '')
        
        # Always proceed even if game is scheduled
        game_id = game['gamePk']
        
        # Use v1.1 endpoint for comprehensive data
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        async with session.get(url, timeout=10) as res:
            if res.status == 404:
                logging.warning("Lineup data not yet available")
                return None
            live_data = await res.json()

        # Check both gameData and liveData sections
        game_data = live_data.get('gameData', {})
        lineup_data = game_data.get('lineups', {})
        if not lineup_data:
            lineup_data = live_data.get('liveData', {}).get('lineups', {})

        home_id = game_data.get('teams', {}).get('home', {}).get('id')
        team_type = 'home' if team_id == home_id else 'away'

        batting_order = lineup_data.get(team_type, {}).get('battingOrder', [])
        players = lineup_data.get(team_type, {}).get('players', {})

        lineup = []
        for player_ref in batting_order:
            player_id = str(player_ref).replace('ID', '')
            player = players.get(player_id)
            if player and 'name' in player.get('person', {}):
                lineup.append(player['person']['name']['fullName'])

        return lineup if validate_lineup(lineup) else None

    except Exception as e:
        logging.error(f"MLB lineup error: {str(e)}", exc_info=True)
        return None
async def try_scrape_lineup(session: aiohttp.ClientSession, team_id: int) -> Optional[List[str]]:
    try:
        return await get_mlb_lineup(session, team_id)
    except Exception:
        return None

async def get_daily_lineup(session: aiohttp.ClientSession, team_name: str) -> List[Dict]:
    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        logging.error(f"Unknown team {team_name}")
        return []

    lineup = await try_scrape_lineup(session, team_id)
    if lineup:
        return [{"name": name} for name in lineup]
    else:
        logging.warning(f"Lineup fallback for {team_name}")
        return [{"name": "Generic Player"} for _ in range(9)]

# --- Strikeout projection ---

def calculate_vulnerability(batters: List[Dict], pitch_type: str) -> np.ndarray:
    weights = {'k_pct': 0.5, 'whiff_pct': 0.4, 'xwoba': 0.1}
    df = pd.DataFrame(batters)
    z_k = ((df['k_pct'] - LEAGUE_AVG['k_pct']) / 0.06).clip(-2.5, 2.5)
    z_whiff = ((df['whiff_pct'] - LEAGUE_AVG['whiff_pct']) / 0.08).clip(-2.5, 2.5)
    z_xwoba = ((LEAGUE_AVG['xwoba'] - df['xwoba']) / 0.04).clip(-2.5, 2.5)
    return np.tanh(z_k*weights['k_pct'] + z_whiff*weights['whiff_pct'] + z_xwoba*weights['xwoba'])*2

def get_platoon_modifier(pitcher_hand: str, batter_hand: str) -> float:
    matchups = {('L','L'):0.93, ('L','R'):1.07, ('R','R'):0.95, ('R','L'):1.05}
    return matchups.get((pitcher_hand,batter_hand),1.0)

def get_park_modifier(park: str) -> float:
    effects = {'Coors Field':0.95, 'Great American Ball Park':1.05, 'T-Mobile Park':1.03, 'Oracle Park':0.97}
    return effects.get(park, 1.0)

def get_team_modifier(team: str, pitcher_hand: str) -> float:
    # Implement your team trends logic here
    return 1.0

def get_dynamic_weights(pitcher_hand: str, pitch_types: List[str]) -> List[float]:
    if 'SL' in pitch_types and 'FF' in pitch_types:
        return [0.5,0.25,0.15,0.1]
    if 'CU' in pitch_types:
        return [0.35,0.35,0.2,0.1]
    if pitcher_hand == 'L':
        return [0.45,0.25,0.2,0.1]
    return [0.4,0.3,0.2,0.1]

def get_pitcher_dispersion(ks_logs: List[int]) -> float:
    return 1.5 if len(ks_logs)<5 else max(1.0,np.std(ks_logs)*0.75)

def simulate_ks(mean: float, dispersion: float, n: int=10000) -> np.ndarray:
    shape = (mean**2) / (dispersion**2)
    scale = (dispersion**2) / mean
    return np.random.gamma(shape, scale, n).clip(0,15)

def preprocess_batter(batter: Dict) -> Dict:
    return {
        'k_pct': batter.get('k_pct', LEAGUE_AVG['k_pct']),
        'whiff_pct': batter.get('whiff_pct', LEAGUE_AVG['whiff_pct']),
        'xwoba': batter.get('xwoba', LEAGUE_AVG['xwoba']),
        'hand': batter.get('hand', 'R')
    }

def project_strikeouts(
    pitcher: str,
    pitcher_hand: str,
    base_ks: float,
    ks_logs: List[int],
    putaway_pitch: str,
    opponent_team: str,
    park: str,
    opponent_lineup: List[Dict]
) -> Dict:
    batters = [preprocess_batter(b) for b in opponent_lineup]
    matchup_scores = calculate_vulnerability(batters, putaway_pitch)
    matchup_mod = 1 + (np.mean(matchup_scores)/10)
    platoon_mod = np.mean([get_platoon_modifier(pitcher_hand, b['hand']) for b in batters])
    park_mod = get_park_modifier(park)
    team_mod = get_team_modifier(opponent_team, pitcher_hand)
    weights = get_dynamic_weights(pitcher_hand, [putaway_pitch])

    total_mod = 1 + sum((
        (matchup_mod - 1)*weights[0],
        (platoon_mod - 1)*weights[1],
        (park_mod - 1)*weights[2],
        (team_mod - 1)*weights[3]
    ))
    total_mod = np.clip(total_mod, 0.85, 1.15)

    # Async Vegas line fetcher — simplified here for demo
    vegas_line, _ = 7.0, 'manual-fallback'

    vegas_effect = 1 / (1 + math.exp(-(vegas_line - 6.5)))
    adjusted_mean = base_ks * total_mod * (0.95 + 0.1 * vegas_effect)

    dispersion = get_pitcher_dispersion(ks_logs)
    samples = simulate_ks(adjusted_mean, dispersion)

    return {
        'pitcher': pitcher,
        'mean': round(adjusted_mean, 2),
        'distribution': {
            '25th': np.percentile(samples, 25),
            '50th': np.percentile(samples, 50),
            '75th': np.percentile(samples, 75),
            '95th': np.percentile(samples, 95)
        },
        'prob_over_5.5': round(np.mean(samples > 5.5)*100, 2),
        'prob_over_6.5': round(np.mean(samples > 6.5)*100, 2),
        'prob_over_7.5': round(np.mean(samples > 7.5)*100, 2)
    }

# Sample game logs for demo
def get_pitcher_game_logs(pitcher_name: str, games: int=15) -> List[int]:
    return [5, 6, 7, 4, 6, 5, 7, 6, 6, 5, 6, 7, 5, 5, 6]

def calculate_ewma(k_logs: List[int], alpha: float=0.3) -> float:
    if not k_logs:
        return LEAGUE_AVG_K
    weights = np.array([(1-alpha)**i for i in range(len(k_logs))][::-1])
    return round(np.dot(k_logs, weights)/weights.sum(), 2)

async def auto_project_strikeouts(pitcher_name: str, opponent_team: str, park: str="PNC Park") -> Optional[Dict]:
    pitcher_id = PITCHER_IDS.get(pitcher_name)
    ks_logs = get_pitcher_game_logs(pitcher_name)
    base_ks = calculate_ewma(ks_logs)
    pitcher_hand = 'R'  # Replace with lookup if available
    putaway_pitch = 'SL'  # Replace with lookup if available

    async with aiohttp.ClientSession() as session:
        lineup_raw = await get_daily_lineup(session, opponent_team)
        opponent_lineup = []
        for batter in lineup_raw:
            name = batter.get('name') if isinstance(batter, dict) else batter
            stats = await get_batter_stats(session, name)
            opponent_lineup.append(stats)

        projection = project_strikeouts(
            pitcher=pitcher_name,
            pitcher_hand=pitcher_hand,
            base_ks=base_ks,
            ks_logs=ks_logs,
            putaway_pitch=putaway_pitch,
            opponent_team=opponent_team,
            park=park,
            opponent_lineup=opponent_lineup
        )
        return projection

# Run example
async def run_single_projection(pitcher, opponent, park):
    proj = await auto_project_strikeouts(pitcher, opponent, park)
    if proj:
        print(f"Projection for {pitcher} vs {opponent} at {park}:")
        print(json.dumps(proj, indent=2))
    else:
        print("Projection failed.")


async def main():
    test_games = [
        {"pitcher": "Zac Gallen", "opponent": "Mets", "park": "Citi Field"},
        {"pitcher": "Max Scherzer", "opponent": "Yankees", "park": "Yankee Stadium"},
        {"pitcher": "Nick Lodolo", "opponent": "Brewers", "park": "American Family Field"},
    ]
    for game in test_games:
        proj = await auto_project_strikeouts(game['pitcher'], game['opponent'], game['park'])
        print(f"\nProjection for {game['pitcher']} vs {game['opponent']} at {game['park']}:")
        print(json.dumps(proj, indent=2))

async def cli():
    if len(sys.argv) < 3:
        print("Usage: python projection.py <pitcher_name> <opponent_team> [park]")
        return
    pitcher = sys.argv[1]
    opponent = sys.argv[2]
    park = sys.argv[3] if len(sys.argv) > 3 else "PNC Park"
    
    proj = await auto_project_strikeouts(pitcher, opponent, park)
    if proj:
        print(json.dumps(proj, indent=2))
    else:
        print("Projection failed.")

async def run_single_projection(pitcher, opponent, park):
    proj = await auto_project_strikeouts(pitcher, opponent, park)
    if proj:
        print(f"\nProjection for {pitcher} vs {opponent} at {park}:")
        print(json.dumps(proj, indent=2))
    else:
        print(f"Projection failed for {pitcher} vs {opponent}")

async def run_all():
    await run_all_projections()

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pitcher = sys.argv[1]
        opponent = sys.argv[2]
        park = sys.argv[3] if len(sys.argv) > 3 else "PNC Park"
        asyncio.run(run_single_projection(pitcher, opponent, park))
    else:
        asyncio.run(run_all())
