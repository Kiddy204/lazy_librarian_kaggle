"""
Lazy Librarian - Book Recommendation System UI
A beautiful Streamlit interface for book recommendations
"""

import streamlit as st
import pandas as pd
import random
import re
import requests
import html
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_preprocessing.config import INTERACTIONS_FILE, ITEMS_FILE
from models.cmf_recommender import CMFRecommender

# =============================================================================
# BOOK COVER API CONFIGURATION
# =============================================================================

# Open Library Covers API - Free, no auth required, direct image URLs
# Format: https://covers.openlibrary.org/b/isbn/{ISBN}-{S|M|L}.jpg
OPENLIBRARY_COVER_URL = "https://covers.openlibrary.org/b/isbn/{isbn}-{size}.jpg"

# Fallback placeholder - using data URI (no external request needed)
# Note: Single quotes are URL-encoded as %27 to avoid breaking HTML attributes
DEFAULT_COVER = "data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%27128%27 height=%27192%27 viewBox=%270 0 128 192%27%3E%3Crect fill=%27%231e3a5f%27 width=%27128%27 height=%27192%27/%3E%3Ctext x=%2764%27 y=%2796%27 font-size=%2748%27 text-anchor=%27middle%27 dominant-baseline=%27middle%27%3E📚%3C/text%3E%3Ctext x=%2764%27 y=%27140%27 font-size=%2710%27 fill=%27%2300d9ff%27 text-anchor=%27middle%27%3ENo Cover%3C/text%3E%3C/svg%3E"

# Cover cache file path
COVER_CACHE_FILE = Path(__file__).parent / "cover_cache.csv"

# =============================================================================
# COVER CACHE FUNCTIONS
# =============================================================================

@st.cache_data
def load_cover_cache() -> dict:
    """
    Load the cover cache from CSV file.
    Returns a dictionary mapping item_id to cached cover URL.
    """
    if not COVER_CACHE_FILE.exists():
        return {}
    
    try:
        cache_df = pd.read_csv(COVER_CACHE_FILE)
        # Create dict: item_id -> {size: url}
        cache = {}
        for _, row in cache_df.iterrows():
            item_id = row['item_id']
            if item_id not in cache:
                cache[item_id] = {}
            cache[item_id][row['size']] = row['cover_url']
        return cache
    except Exception as e:
        st.warning(f"Could not load cover cache: {e}")
        return {}


def save_cover_to_cache(item_id: int, isbn: str, cover_url: str, size: str = "M"):
    """
    Save a working cover URL to the cache file.
    """
    # Load existing cache
    if COVER_CACHE_FILE.exists():
        try:
            cache_df = pd.read_csv(COVER_CACHE_FILE)
        except:
            cache_df = pd.DataFrame(columns=['item_id', 'isbn', 'cover_url', 'size'])
    else:
        cache_df = pd.DataFrame(columns=['item_id', 'isbn', 'cover_url', 'size'])
    
    # Check if already cached for this size
    existing = cache_df[(cache_df['item_id'] == item_id) & (cache_df['size'] == size)]
    if len(existing) > 0:
        # Update existing entry
        cache_df.loc[(cache_df['item_id'] == item_id) & (cache_df['size'] == size), 'cover_url'] = cover_url
        cache_df.loc[(cache_df['item_id'] == item_id) & (cache_df['size'] == size), 'isbn'] = isbn
    else:
        # Add new entry
        new_row = pd.DataFrame([{
            'item_id': item_id,
            'isbn': isbn,
            'cover_url': cover_url,
            'size': size
        }])
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    
    # Save to file
    cache_df.to_csv(COVER_CACHE_FILE, index=False)

    # Clear the Streamlit cache to reload fresh data
    load_cover_cache.clear()


def verify_cover_url(url: str, timeout: float = 5.0) -> bool:
    """
    Verify if a cover URL returns a valid image (not a 1x1 placeholder).
    Open Library returns a 1x1 transparent GIF for missing covers.
    """
    if url.startswith("data:"):
        return False  # It's our placeholder
    
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code != 200:
            return False
        
        # Check content length - 1x1 GIF is very small (< 100 bytes)
        content_length = response.headers.get('Content-Length', '0')
        if int(content_length) < 1000:  # Real covers are > 1KB
            return False
        
        return True
    except:
        return False


def find_working_cover(isbn_string: str, size: str = "M") -> str | None:
    """
    Find a working cover URL by testing each ISBN.
    Returns the first working URL or None if no cover found.
    """
    isbns = extract_all_isbns(isbn_string)
    
    for isbn in isbns:
        url = OPENLIBRARY_COVER_URL.format(isbn=isbn, size=size)
        if verify_cover_url(url):
            return url, isbn
    
    return None, None


def build_cover_cache(items: pd.DataFrame, progress_callback=None, size: str = "M") -> int:
    """
    Build/update the cover cache by testing all items.
    Returns the number of covers found.
    """
    cache = load_cover_cache()
    found_count = 0
    
    for idx, (_, item) in enumerate(items.iterrows()):
        item_id = item['i']
        isbn_string = item.get('ISBN Valid', '')
        
        # Skip if already cached for this size
        if item_id in cache and size in cache[item_id]:
            found_count += 1
            if progress_callback:
                progress_callback(idx + 1, len(items), found_count)
            continue
        
        # Try to find a working cover
        working_url, working_isbn = find_working_cover(isbn_string, size)
        
        if working_url:
            save_cover_to_cache(item_id, working_isbn, working_url, size)
            found_count += 1
        
        if progress_callback:
            progress_callback(idx + 1, len(items), found_count)
    
    # Clear the cache to reload fresh data
    load_cover_cache.clear()
    
    return found_count


def extract_all_isbns(isbn_string: str) -> list[str]:
    """
    Extract ALL valid ISBNs from the ISBN Valid column.
    ISBNs can be ISBN-10 (10 digits) or ISBN-13 (13 digits).
    The column may contain multiple ISBNs separated by semicolons.
    
    Returns list of valid ISBNs, prioritizing ISBN-13 over ISBN-10.
    """
    if pd.isna(isbn_string) or not isbn_string:
        return []
    
    isbns_13 = []
    isbns_10 = []
    
    # Split by semicolon and try each ISBN
    isbn_candidates = str(isbn_string).split(';')
    
    for candidate in isbn_candidates:
        # Clean up the candidate - remove all non-digit characters except X
        cleaned = re.sub(r'[^0-9X]', '', candidate.strip().upper())
        
        # Check if it's a valid ISBN-13 (13 digits)
        if len(cleaned) == 13 and cleaned.isdigit():
            if cleaned not in isbns_13:
                isbns_13.append(cleaned)
        # Check if it's a valid ISBN-10 (10 chars, may end with X)
        elif len(cleaned) == 10:
            if cleaned not in isbns_10:
                isbns_10.append(cleaned)
    
    # Return ISBN-13s first (better coverage), then ISBN-10s
    return isbns_13 + isbns_10


def get_cover_urls(isbn_string: str, size: str = "M", item_id: int = None) -> list[str]:
    """
    Get cover URLs with cache support.

    If item_id is provided and a cached URL exists, returns cached URL first.
    Otherwise tries each ISBN in the field with Open Library API.
    Returns a list of URLs to try in order, with placeholder as last resort.
    """
    urls = []

    # Check cache first if item_id provided
    if item_id is not None:
        cache = load_cover_cache()
        if item_id in cache and size in cache[item_id]:
            # Cached URL goes first
            urls.append(cache[item_id][size])

    # Add Open Library URLs for all ISBNs
    isbns = extract_all_isbns(isbn_string)
    for isbn in isbns:
        url = OPENLIBRARY_COVER_URL.format(isbn=isbn, size=size)
        if url not in urls:  # Avoid duplicates if cached URL is same
            urls.append(url)

    # Add default placeholder as last resort
    if not urls:
        return [DEFAULT_COVER]

    urls.append(DEFAULT_COVER)

    return urls


def get_cover_url(isbn_string: str, size: str = "M", item_id: int = None) -> str:
    """
    Get the primary cover URL for a book.
    For simple usage when fallback JS is handling errors.
    """
    urls = get_cover_urls(isbn_string, size, item_id)
    return urls[0] if urls else DEFAULT_COVER

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="📚 Lazy Librarian",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b69 0%, #1a1a2e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e8e8e8;
    }
    
    /* Card styling */
    .book-card {
        background: linear-gradient(145deg, #1e3a5f, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.3);
    }
    
    .user-card {
        background: linear-gradient(145deg, #2d1b69, #1a1a2e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d9ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .user-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.3);
    }
    
    .recommendation-card {
        background: linear-gradient(145deg, #0d7377, #14a3a8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ffd700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #e94560, #ff6b6b);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    .metric-card-blue {
        background: linear-gradient(145deg, #00d9ff, #0099cc);
    }
    
    .metric-card-purple {
        background: linear-gradient(145deg, #8b5cf6, #7c3aed);
    }
    
    .metric-card-green {
        background: linear-gradient(145deg, #10b981, #059669);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #00d9ff, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d9ff;
        border-bottom: 2px solid #e94560;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff6b6b, #e94560);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.5);
        transform: translateY(-2px);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Search box */
    .stTextInput > div > div > input {
        background-color: #1e3a5f;
        border: 2px solid #00d9ff;
        border-radius: 10px;
        color: white;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1e3a5f;
        border: 2px solid #00d9ff;
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #e94560, #00d9ff);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e3a5f;
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-subject {
        background: linear-gradient(90deg, #8b5cf6, #7c3aed);
        color: white;
    }
    
    .badge-count {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
    }
    
    /* Book cover styling */
    .book-card-with-cover {
        background: linear-gradient(145deg, #1e3a5f, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        gap: 20px;
    }
    
    .book-card-with-cover:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.3);
    }
    
    .book-cover {
        flex-shrink: 0;
        width: 100px;
        height: 150px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        background: linear-gradient(145deg, #2d1b69, #1a1a2e);
    }
    
    .book-cover img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .book-cover-placeholder {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #2d1b69, #1a1a2e);
        color: #666;
        font-size: 2rem;
    }
    
    .book-details {
        flex: 1;
        min-width: 0;
    }
    
    .recommendation-card-with-cover {
        background: linear-gradient(145deg, #0d7377, #14a3a8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ffd700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        display: flex;
        gap: 15px;
    }
    
    .recommendation-cover {
        flex-shrink: 0;
        width: 80px;
        height: 120px;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 3px 8px rgba(0,0,0,0.3);
    }
    
    .recommendation-cover img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load interactions and items data."""
    try:
        interactions = pd.read_csv(INTERACTIONS_FILE)
        items = pd.read_csv(ITEMS_FILE)
        return interactions, items
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, None


@st.cache_resource
def load_recommender():
    """Load the CMF recommender model (cached)."""
    return CMFRecommender()


def get_user_stats(interactions: pd.DataFrame) -> pd.DataFrame:
    """Get statistics for each user."""
    user_stats = interactions.groupby('u').agg(
        total_borrows=('i', 'count'),
        unique_books=('i', 'nunique'),
        first_borrow=('t', 'min'),
        last_borrow=('t', 'max')
    ).reset_index()
    user_stats.columns = ['User ID', 'Total Borrows', 'Unique Books', 'First Borrow', 'Last Borrow']
    return user_stats.sort_values('Total Borrows', ascending=False)


def get_user_history(interactions: pd.DataFrame, items: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """Get borrowing history for a specific user."""
    user_interactions = interactions[interactions['u'] == user_id]
    borrow_counts = user_interactions.groupby('i').size().reset_index(name='Times Borrowed')
    
    # Merge with item details
    history = borrow_counts.merge(items, left_on='i', right_on='i', how='left')
    history = history.sort_values('Times Borrowed', ascending=False)
    return history


def get_recommendations(items: pd.DataFrame, exclude_items: list, n: int = 10, user_id: int = None) -> pd.DataFrame:
    """
    Get book recommendations for a user using the trained CMF model.
    Falls back to popular items if model is not available.
    
    Args:
        items: DataFrame of all items
        exclude_items: List of item IDs to exclude (already borrowed)
        n: Number of recommendations to return
        user_id: User ID to get recommendations for
        
    Returns:
        DataFrame of recommended items
    """
    recommender = load_recommender()
    
    # Try to use the CMF model if available and user_id is provided
    if user_id is not None:
        try:
            # Get recommendations from the model (it handles exclusions internally)
            recommended_item_ids = recommender.get_recommendations_for_user(
                user_id,
                n_recommendations=n,
                exclude_items=exclude_items
            )
            
            if recommended_item_ids:
                # Get the item details
                recommendations = items[items['i'].isin(recommended_item_ids)]
                
                # Preserve the order from the model
                if len(recommendations) > 0:
                    recommendations = recommendations.set_index('i').loc[
                        [item_id for item_id in recommended_item_ids if item_id in recommendations.index]
                    ].reset_index()
                    
                    return recommendations
            
        except Exception as e:
            st.warning(f"Error getting model recommendations: {e}. Using fallback.")
    
    # Fallback: popular items
    available_items = items[~items['i'].isin(exclude_items)]
    if len(available_items) == 0:
        return pd.DataFrame()
    
    n_recommendations = min(n, len(available_items))
    
    # Get popular items based on borrow count if available
    try:
        interactions = pd.read_csv(INTERACTIONS_FILE)
        borrow_counts = interactions.groupby('i').size().reset_index(name='count')
        available_with_counts = available_items.merge(borrow_counts, on='i', how='left')
        available_with_counts['count'] = available_with_counts['count'].fillna(0)
        recommendations = available_with_counts.nlargest(n_recommendations, 'count')
    except:
        # If that fails, just sample randomly
        recommendations = available_items.sample(n=n_recommendations)
    
    return recommendations


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-title">📚 Lazy Librarian</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p style="text-align: center; color: #a8a8a8; font-size: 1.1rem; margin-bottom: 30px;">
            Intelligent Book Recommendation System
        </p>
    """, unsafe_allow_html=True)


def render_metrics(interactions: pd.DataFrame, items: pd.DataFrame):
    """Render overview metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; font-size: 2.5rem;">👥 {interactions['u'].nunique():,}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Users</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <h2 style="margin: 0; font-size: 2.5rem;">📖 {len(items):,}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Books</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card metric-card-purple">
                <h2 style="margin: 0; font-size: 2.5rem;">🔄 {len(interactions):,}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Borrows</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_borrows = len(interactions) / interactions['u'].nunique()
        st.markdown(f"""
            <div class="metric-card metric-card-green">
                <h2 style="margin: 0; font-size: 2.5rem;">📊 {avg_borrows:.1f}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Avg. Borrows/User</p>
            </div>
        """, unsafe_allow_html=True)


def render_book_card(book: pd.Series, times_borrowed: int = None):
    """Render a single book card with cover image and fallback support."""
    # Get and escape all text content to prevent HTML injection
    title = html.escape(str(book.get('Title', 'Unknown Title')))
    author = html.escape(str(book.get('Author', 'Unknown Author'))) if pd.notna(book.get('Author')) else 'Unknown Author'
    publisher = html.escape(str(book.get('Publisher', 'Unknown Publisher'))) if pd.notna(book.get('Publisher')) else 'Unknown Publisher'
    subjects = str(book.get('Subjects', ''))
    subjects = re.sub(r'<[^>]+>', '', subjects).strip()
    isbn = book.get('ISBN Valid', '')
    item_id = book.get('i', None)
    
    # Get cover URLs for fallback (with cache support)
    cover_urls = get_cover_urls(isbn, size="M", item_id=item_id)
    # Properly encode as JSON and escape for HTML attribute
    cover_urls_json = html.escape(json.dumps(cover_urls))
    primary_url = cover_urls[0] if cover_urls else DEFAULT_COVER
    
    # Parse and escape subjects
    subject_badges = ""
    if pd.notna(subjects) and subjects:
        subject_list = str(subjects).split(';')[:3]  # Show max 3 subjects
        for subj in subject_list:
            escaped_subj = html.escape(subj.strip())
            subject_badges += f'<span class="badge badge-subject">{escaped_subj}</span> '
    
    borrow_badge = ""
    if times_borrowed:
        borrow_badge = f'<span class="badge badge-count">Borrowed {times_borrowed}x</span>'
    
    # Truncate long titles (after escaping)
    display_title = title[:60] + '...' if len(title) > 60 else title
    
    st.markdown(f"""
        <div class="book-card-with-cover">
            <div class="book-cover">
                <img src="{primary_url}" alt="Book cover"
                     data-fallback-urls="{cover_urls_json}"
                     data-fallback-index="0"
                     onerror="
                        var urls = JSON.parse(this.dataset.fallbackUrls);
                        var idx = parseInt(this.dataset.fallbackIndex) + 1;
                        if (idx < urls.length) {{
                            this.dataset.fallbackIndex = idx;
                            this.src = urls[idx];
                        }} else {{
                            this.onerror = null;
                            this.parentElement.innerHTML = '&lt;div class=book-cover-placeholder&gt;📚&lt;/div&gt;';
                        }}
                     ">
            </div>
            <div class="book-details">
                <h3 style="color: #ffffff; margin: 0 0 8px 0; font-size: 1.1rem;">{display_title}</h3>
                <p style="color: #00d9ff; margin: 4px 0; font-size: 0.9rem;">✍️ {author}</p>
                <p style="color: #a8a8a8; margin: 4px 0; font-size: 0.85rem;">🏢 {publisher}</p>
                <div style="margin-top: 8px;">
                    {borrow_badge}
                </div>
                <div style="margin-top: 5px; overflow: hidden;">
                    {subject_badges}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(book: pd.Series, score: float = None, borrow_count: int = None):
    """Render a recommendation card with cover image, subjects, and borrow count."""
    # Escape all text content
    title = html.escape(str(book.get('Title', 'Unknown Title')))
    author = html.escape(str(book.get('Author', 'Unknown Author'))) if pd.notna(book.get('Author')) else 'Unknown'
    isbn = book.get('ISBN Valid', '')
    item_id = book.get('i', None)
    
    # Extract and clean subjects
    subjects = str(book.get('Subjects', ''))
    subjects = re.sub(r'<[^>]+>', '', subjects).strip()
    
    # Get cover URLs for fallback (smaller size for recommendations, with cache)
    cover_urls = get_cover_urls(isbn, size="S", item_id=item_id)
    cover_urls_json = html.escape(json.dumps(cover_urls))
    primary_url = cover_urls[0] if cover_urls else DEFAULT_COVER
    
    # Build subject badges
    subject_badges = ""
    if pd.notna(subjects) and subjects:
        subject_list = str(subjects).split(';')[:3]  # Show max 3 subjects
        for subj in subject_list:
            escaped_subj = html.escape(subj.strip())
            subject_badges += f'<span class="badge badge-subject">{escaped_subj}</span> '
    
    # Build borrow count badge
    borrow_badge = ""
    if borrow_count and borrow_count > 0:
        borrow_badge = f'<span class="badge badge-count">Borrowed {borrow_count}x</span>'
    
    score_text = ""
    if score:
        score_text = f'<div style="color: #ffd700; font-weight: bold; margin-top: 5px;">⭐ {score:.2f} match</div>'
    
    # Truncate long titles (after escaping)
    display_title = title[:45] + '...' if len(title) > 45 else title
    
    st.markdown(f"""
        <div class="recommendation-card-with-cover">
            <div class="recommendation-cover">
                <img src="{primary_url}" alt="Book cover"
                     data-fallback-urls="{cover_urls_json}"
                     data-fallback-index="0"
                     onerror="
                        var urls = JSON.parse(this.dataset.fallbackUrls);
                        var idx = parseInt(this.dataset.fallbackIndex) + 1;
                        if (idx < urls.length) {{
                            this.dataset.fallbackIndex = idx;
                            this.src = urls[idx];
                        }} else {{
                            this.onerror = null;
                            this.style.display = 'none';
                            this.parentElement.innerHTML = '&lt;div class=rec-placeholder&gt;📚&lt;/div&gt;';
                        }}
                     ">
            </div>
            <div style="flex: 1; min-width: 0;">
                <h4 style="color: #ffffff; margin: 0; font-size: 1rem;">{display_title}</h4>
                <p style="color: #e8e8e8; margin: 5px 0 0 0; font-size: 0.9rem;">by {author}</p>
                <div style="margin-top: 8px;">
                    {borrow_badge}
                </div>
                <div style="margin-top: 5px; overflow: hidden;">
                    {subject_badges}
                </div>
                {score_text}
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGES
# =============================================================================

def page_home(interactions: pd.DataFrame, items: pd.DataFrame):
    """Home page with overview."""
    render_header()
    st.markdown("---")
    render_metrics(interactions, items)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-title">🏆 Most Active Users</h2>', unsafe_allow_html=True)
        top_users = get_user_stats(interactions).head(5)
        for _, user in top_users.iterrows():
            st.markdown(f"""
                <div class="user-card">
                    <h4 style="color: #00d9ff; margin: 0;">👤 User {int(user['User ID'])}</h4>
                    <p style="color: #a8a8a8; margin: 5px 0;">
                        📚 {int(user['Total Borrows'])} borrows | 
                        📖 {int(user['Unique Books'])} unique books
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-title">📚 Most Borrowed Books</h2>', unsafe_allow_html=True)
        book_counts = interactions.groupby('i').size().reset_index(name='count')
        top_books = book_counts.nlargest(5, 'count')
        
        for _, row in top_books.iterrows():
            book_data = items[items['i'] == row['i']]
            if len(book_data) > 0:
                book = book_data.iloc[0]
                raw_title = str(book.get('Title', 'Unknown'))
                title = html.escape(raw_title[:50] + ('...' if len(raw_title) > 50 else ''))
                isbn = book.get('ISBN Valid', '')
                item_id = row['i']
                
                # Get cover URLs for fallback (with cache support)
                cover_urls = get_cover_urls(isbn, size="S", item_id=item_id)
                cover_urls_json = html.escape(json.dumps(cover_urls))
                primary_url = cover_urls[0] if cover_urls else DEFAULT_COVER
                
                st.markdown(f"""
                    <div class="book-card-with-cover" style="padding: 15px;">
                        <div class="book-cover" style="width: 60px; height: 90px;">
                            <img src="{primary_url}" alt="Cover"
                                 data-fallback-urls="{cover_urls_json}"
                                 data-fallback-index="0"
                                 onerror="
                                    var urls = JSON.parse(this.dataset.fallbackUrls);
                                    var idx = parseInt(this.dataset.fallbackIndex) + 1;
                                    if (idx < urls.length) {{
                                        this.dataset.fallbackIndex = idx;
                                        this.src = urls[idx];
                                    }} else {{
                                        this.onerror = null;
                                        this.parentElement.innerHTML = '&lt;div class=book-cover-placeholder&gt;📚&lt;/div&gt;';
                                    }}
                                 ">
                        </div>
                        <div style="flex: 1;">
                            <h4 style="color: #ffffff; margin: 0; font-size: 0.95rem;">{title}</h4>
                            <p style="color: #e94560; margin: 5px 0; font-size: 0.9rem;">
                                <strong>{int(row['count'])}</strong> times borrowed
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)


def page_users(interactions: pd.DataFrame, items: pd.DataFrame):
    """Users listing page with pagination."""
    st.markdown('<h1 class="main-title">👥 Users</h1>', unsafe_allow_html=True)
    
    # Initialize pagination state
    if 'users_page' not in st.session_state:
        st.session_state.users_page = 0
    
    # Search and filter
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_user = st.text_input("🔍 Search User ID", placeholder="Enter user ID...", key="user_search")
    with col2:
        sort_by = st.selectbox("Sort by", ["Total Borrows", "Unique Books", "User ID"])
    with col3:
        items_per_page = st.selectbox("Per page", [12, 24, 48, 96], index=1)
    
    user_stats = get_user_stats(interactions)
    
    # Apply search filter (reset page when searching)
    if search_user:
        user_stats = user_stats[user_stats['User ID'].astype(str).str.contains(search_user)]
        st.session_state.users_page = 0
    
    # Apply sorting
    sort_map = {
        "Total Borrows": "Total Borrows",
        "Unique Books": "Unique Books", 
        "User ID": "User ID"
    }
    ascending = sort_by == "User ID"
    user_stats = user_stats.sort_values(sort_map[sort_by], ascending=ascending)
    
    # Calculate pagination
    total_users = len(user_stats)
    total_pages = max(1, (total_users + items_per_page - 1) // items_per_page)
    
    # Ensure current page is valid
    if st.session_state.users_page >= total_pages:
        st.session_state.users_page = total_pages - 1
    if st.session_state.users_page < 0:
        st.session_state.users_page = 0
    
    current_page = st.session_state.users_page
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_users)
    
    # Display info and pagination controls
    st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_users} users**")
    
    # Pagination controls at top
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0, key="users_first"):
            st.session_state.users_page = 0
            st.rerun()
    with col2:
        if st.button("◀️ Prev", disabled=current_page == 0, key="users_prev"):
            st.session_state.users_page -= 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; color: #00d9ff; font-weight: bold;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1, key="users_next"):
            st.session_state.users_page += 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1, key="users_last"):
            st.session_state.users_page = total_pages - 1
            st.rerun()
    
    st.markdown("---")
    
    # Get current page data
    page_users = user_stats.iloc[start_idx:end_idx]
    
    # Display users in a grid
    cols = st.columns(3)
    for idx, (_, user) in enumerate(page_users.iterrows()):
        with cols[idx % 3]:
            if st.button(
                f"👤 User {int(user['User ID'])}\n📚 {int(user['Total Borrows'])} borrows",
                key=f"user_{user['User ID']}_{current_page}",
                use_container_width=True
            ):
                st.session_state.selected_user = int(user['User ID'])
                st.session_state.page = "user_detail"
                st.rerun()
    
    # Pagination controls at bottom
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0, key="users_first_bottom"):
            st.session_state.users_page = 0
            st.rerun()
    with col2:
        if st.button("◀️ Prev", disabled=current_page == 0, key="users_prev_bottom"):
            st.session_state.users_page -= 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; color: #00d9ff; font-weight: bold;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1, key="users_next_bottom"):
            st.session_state.users_page += 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1, key="users_last_bottom"):
            st.session_state.users_page = total_pages - 1
            st.rerun()


def page_items(interactions: pd.DataFrame, items: pd.DataFrame):
    """Items/Books listing page with pagination."""
    st.markdown('<h1 class="main-title">📚 Books Catalog</h1>', unsafe_allow_html=True)
    
    # Initialize pagination state
    if 'books_page' not in st.session_state:
        st.session_state.books_page = 0
    
    # Search and filter
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search_title = st.text_input("🔍 Search by Title", placeholder="Enter book title...", key="book_title_search")
    with col2:
        search_author = st.text_input("✍️ Filter by Author", placeholder="Author name...", key="book_author_search")
    with col3:
        sort_by = st.selectbox("Sort by", ["Borrow Count", "Title", "Author"], key="book_sort")
    with col4:
        items_per_page = st.selectbox("Per page", [10, 20, 40, 80], index=1, key="book_per_page")
    
    # Filter items
    filtered_items = items.copy()
    search_changed = False
    
    if search_title:
        filtered_items = filtered_items[
            filtered_items['Title'].str.lower().str.contains(search_title.lower(), na=False)
        ]
        search_changed = True
    if search_author:
        filtered_items = filtered_items[
            filtered_items['Author'].str.lower().str.contains(search_author.lower(), na=False)
        ]
        search_changed = True
    
    # Reset page when search changes
    if search_changed:
        st.session_state.books_page = 0
    
    # Add borrow counts
    borrow_counts = interactions.groupby('i').size().reset_index(name='Borrow Count')
    filtered_items = filtered_items.merge(borrow_counts, on='i', how='left')
    filtered_items['Borrow Count'] = filtered_items['Borrow Count'].fillna(0).astype(int)
    
    # Apply sorting
    if sort_by == "Borrow Count":
        filtered_items = filtered_items.sort_values('Borrow Count', ascending=False)
    elif sort_by == "Title":
        filtered_items = filtered_items.sort_values('Title', ascending=True)
    elif sort_by == "Author":
        filtered_items = filtered_items.sort_values('Author', ascending=True, na_position='last')
    
    # Calculate pagination
    total_books = len(filtered_items)
    total_pages = max(1, (total_books + items_per_page - 1) // items_per_page)
    
    # Ensure current page is valid
    if st.session_state.books_page >= total_pages:
        st.session_state.books_page = total_pages - 1
    if st.session_state.books_page < 0:
        st.session_state.books_page = 0
    
    current_page = st.session_state.books_page
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_books)
    
    # Display info
    st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_books} books**")
    
    # Pagination controls at top
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0, key="books_first"):
            st.session_state.books_page = 0
            st.rerun()
    with col2:
        if st.button("◀️ Prev", disabled=current_page == 0, key="books_prev"):
            st.session_state.books_page -= 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; color: #00d9ff; font-weight: bold;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1, key="books_next"):
            st.session_state.books_page += 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1, key="books_last"):
            st.session_state.books_page = total_pages - 1
            st.rerun()
    
    st.markdown("---")
    
    # Get current page data
    page_books = filtered_items.iloc[start_idx:end_idx]
    
    # Display books
    cols = st.columns(2)
    for idx, (_, book) in enumerate(page_books.iterrows()):
        with cols[idx % 2]:
            render_book_card(book, int(book['Borrow Count']) if book['Borrow Count'] > 0 else None)
    
    # Pagination controls at bottom
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0, key="books_first_bottom"):
            st.session_state.books_page = 0
            st.rerun()
    with col2:
        if st.button("◀️ Prev", disabled=current_page == 0, key="books_prev_bottom"):
            st.session_state.books_page -= 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px; color: #00d9ff; font-weight: bold;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1, key="books_next_bottom"):
            st.session_state.books_page += 1
            st.rerun()
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1, key="books_last_bottom"):
            st.session_state.books_page = total_pages - 1
            st.rerun()


def page_user_detail(interactions: pd.DataFrame, items: pd.DataFrame):
    """User detail page with history and recommendations."""
    user_id = st.session_state.get('selected_user', None)
    
    if user_id is None:
        st.warning("No user selected. Please go to the Users page and select a user.")
        return
    
    # Back button
    if st.button("← Back to Users"):
        st.session_state.page = "users"
        st.rerun()
    
    st.markdown(f'<h1 class="main-title">👤 User {user_id}</h1>', unsafe_allow_html=True)
    
    # User stats
    user_interactions = interactions[interactions['u'] == user_id]
    total_borrows = len(user_interactions)
    unique_books = user_interactions['i'].nunique()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; font-size: 2rem;">🔄 {total_borrows}</h2>
                <p style="margin: 5px 0 0 0;">Total Borrows</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <h2 style="margin: 0; font-size: 2rem;">📖 {unique_books}</h2>
                <p style="margin: 5px 0 0 0;">Unique Books</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_per_book = total_borrows / unique_books if unique_books > 0 else 0
        st.markdown(f"""
            <div class="metric-card metric-card-purple">
                <h2 style="margin: 0; font-size: 2rem;">📊 {avg_per_book:.1f}</h2>
                <p style="margin: 5px 0 0 0;">Avg. Borrows/Book</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for history and recommendations
    tab1, tab2 = st.tabs(["📚 Borrowing History", "✨ Recommendations"])
    
    with tab1:
        st.markdown('<h2 class="section-title">📚 Books Borrowed</h2>', unsafe_allow_html=True)
        
        history = get_user_history(interactions, items, user_id)
        
        if len(history) == 0:
            st.info("No borrowing history found for this user.")
        else:
            cols = st.columns(2)
            for idx, (_, book) in enumerate(history.iterrows()):
                with cols[idx % 2]:
                    render_book_card(book, int(book['Times Borrowed']))
    
    with tab2:
        st.markdown('<h2 class="section-title">✨ Recommended for You</h2>', unsafe_allow_html=True)
        
        # Check if model is available
        recommender = load_recommender()
        if recommender.is_available():
            st.success("🤖 **Powered by CMF Model** - Personalized recommendations based on collaborative filtering")
        else:
            st.info("🔮 **Note:** CMF model not available. Showing popular books instead.")
        
        # Get user's borrowed items
        user_items = user_interactions['i'].unique().tolist()
        
        # Get recommendations using the model
        recommendations = get_recommendations(items, user_items, n=10, user_id=user_id)
        
        if len(recommendations) == 0:
            st.warning("No recommendations available.")
        else:
            # Calculate borrow counts for recommended books
            borrow_counts = interactions.groupby('i').size().reset_index(name='borrow_count')
            recommendations = recommendations.merge(borrow_counts, on='i', how='left')
            recommendations['borrow_count'] = recommendations['borrow_count'].fillna(0).astype(int)
            
            cols = st.columns(2)
            for idx, (_, book) in enumerate(recommendations.iterrows()):
                with cols[idx % 2]:
                    # Show rank instead of fake score
                    rank = idx + 1
                    render_recommendation_card(book, score=None, borrow_count=int(book['borrow_count']))
                    st.markdown(f"<div style='text-align: center; color: #ffd700; font-weight: bold; margin-top: -10px;'>#{rank} Recommendation</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    
    # Load data
    interactions, items = load_data()
    
    if interactions is None or items is None:
        st.error("⚠️ Could not load data. Please ensure data files exist in the data/ directory.")
        st.markdown("""
            ### Expected files:
            - `data/interactions.csv` (columns: u, i, t)
            - `data/items.csv` (columns: Title, Author, ISBN Valid, Publisher, Subjects, i)
        """)
        return
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <h2 style="color: #00d9ff;">📚 Lazy Librarian</h2>
                <p style="color: #a8a8a8; font-size: 0.9rem;">Book Recommendations</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("👥 Users", use_container_width=True):
            st.session_state.page = 'users'
            st.rerun()
        
        if st.button("📚 Books", use_container_width=True):
            st.session_state.page = 'items'
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("""
            <div style="padding: 15px; background: rgba(0,217,255,0.1); border-radius: 10px;">
                <h4 style="color: #00d9ff; margin: 0;">📊 Quick Stats</h4>
            </div>
        """, unsafe_allow_html=True)
        
        st.metric("Total Users", f"{interactions['u'].nunique():,}")
        st.metric("Total Books", f"{len(items):,}")
        st.metric("Total Borrows", f"{len(interactions):,}")
        
        st.markdown("---")
        
        # Cover Cache Management
        st.markdown("""
            <div style="padding: 15px; background: rgba(233,69,96,0.1); border-radius: 10px;">
                <h4 style="color: #e94560; margin: 0;">🖼️ Cover Cache</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Show cache stats
        cache = load_cover_cache()
        cached_count = len(cache)
        total_books = len(items)
        cache_percent = (cached_count / total_books * 100) if total_books > 0 else 0
        
        st.markdown(f"**Cached:** {cached_count:,} / {total_books:,} ({cache_percent:.1f}%)")
        
        # Build cache button
        if st.button("🔄 Build Cover Cache", use_container_width=True, help="Scan all books and cache working cover URLs"):
            st.session_state.building_cache = True
            st.rerun()
        
        # Show progress if building
        if st.session_state.get('building_cache', False):
            progress_bar = st.progress(0, text="Scanning covers...")
            status_text = st.empty()
            
            def update_progress(current, total, found):
                progress = current / total
                progress_bar.progress(progress, text=f"Scanning... {current}/{total}")
                status_text.markdown(f"**Found:** {found} covers")
            
            # Build cache with progress updates
            found = build_cover_cache(items, progress_callback=update_progress, size="M")
            
            # Also cache small size for recommendations
            progress_bar.progress(0, text="Caching small sizes...")
            found_small = build_cover_cache(items, progress_callback=update_progress, size="S")
            
            st.session_state.building_cache = False
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Cache built! Found {found} covers.")
            st.rerun()
        
        # Clear cache button
        if cached_count > 0:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                if COVER_CACHE_FILE.exists():
                    os.remove(COVER_CACHE_FILE)
                    load_cover_cache.clear()
                    st.success("Cache cleared!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
                <p>Built with ❤️ using Streamlit</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Render selected page
    if st.session_state.page == 'home':
        page_home(interactions, items)
    elif st.session_state.page == 'users':
        page_users(interactions, items)
    elif st.session_state.page == 'items':
        page_items(interactions, items)
    elif st.session_state.page == 'user_detail':
        page_user_detail(interactions, items)


if __name__ == "__main__":
    main()
