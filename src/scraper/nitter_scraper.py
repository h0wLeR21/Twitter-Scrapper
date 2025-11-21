from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

import time
import csv
import json
import re
import hashlib
import os
from datetime import datetime
from langdetect import detect
from functools import lru_cache


# ---------------------------
# Utility Functions
# ---------------------------

@lru_cache(maxsize=1000)
def hash_username(username):
    """Create a consistent hash for username (cached)."""
    return hashlib.sha256(username.encode()).hexdigest()[:16]


@lru_cache(maxsize=10000)
def detect_language(text):
    """Detect language with caching, default to English on failure."""
    try:
        return detect(text)
    except:
        return 'en'


def parse_date(date_str):
    """Parse Nitter date from 'title' attribute."""
    if not date_str:
        return datetime.now()
    
    try:
        if '·' in date_str:
            date_part, time_part = date_str.split("·")
            date_part = date_part.strip()
            time_part = time_part.strip().replace(" UTC", "")
            full_str = f"{date_part} {time_part}"
            return datetime.strptime(full_str, "%b %d, %Y %I:%M %p")
    except:
        pass
    
    return datetime.now()


@lru_cache(maxsize=10000)
def extract_tweet_id(url):
    """Extract tweet ID from URL (cached)."""
    if not url:
        return None
    match = re.search(r"/status/(\d+)", url)
    return match.group(1) if match else None


def extract_urls(tweet_element):
    """Extract external URLs from tweet (optimized)."""
    urls = []
    try:
        link_elements = tweet_element.find_elements(By.CSS_SELECTOR, ".tweet-content a[href^='http']")
        for link in link_elements:
            if not link.get_attribute("class"):  # Exclude styled links
                href = link.get_attribute("href")
                if href:
                    urls.append(href)
    except:
        pass
    return urls


# Compile regex patterns once for better performance
HASHTAG_PATTERN = re.compile(r'#\w+')

def extract_hashtags(tweet_text):
    """Extract hashtags from tweet text (optimized with compiled regex)."""
    return HASHTAG_PATTERN.findall(tweet_text)


def extract_mentions(tweet_element):
    """Extract @mentions from tweet (optimized)."""
    mentions = []
    try:
        # More specific selector to avoid unnecessary elements
        mention_elements = tweet_element.find_elements(By.CSS_SELECTOR, ".tweet-content a[href^='/'][href*='@']")
        for mention in mention_elements:
            text = mention.text.strip()
            if text.startswith("@"):
                mentions.append(text)
    except:
        pass
    return mentions


def extract_media(container):
    """Extract media information (optimized with single query)."""
    media = []
    try:
        # Images
        images = container.find_elements(By.CSS_SELECTOR, ".attachment.image img")
        media.extend({
            "type": "image",
            "url": img.get_attribute("src") or "",
            "alt": img.get_attribute("alt") or ""
        } for img in images)
        
        # Videos/GIFs - count only
        video_count = len(container.find_elements(By.CSS_SELECTOR, ".attachment.video"))
        media.extend({"type": "video", "url": "", "alt": ""} for _ in range(video_count))
    except:
        pass
    return media


def is_retweet(container):
    """Check if tweet is a retweet (optimized)."""
    try:
        container.find_element(By.CSS_SELECTOR, ".retweet-header")
        return True
    except NoSuchElementException:
        return False


def is_quote_tweet(container):
    """Check if tweet is a quote (optimized)."""
    try:
        container.find_element(By.CSS_SELECTOR, ".quote")
        return True
    except NoSuchElementException:
        return False


def parse_engagement_number(text):
    """Parse engagement count from text (optimized)."""
    if not text:
        return 0
    text = text.strip().lower()
    try:
        if "k" in text:
            return int(float(text.replace("k", "")) * 1000)
        if "m" in text:
            return int(float(text.replace("m", "")) * 1_000_000)
        return int("".join(filter(str.isdigit, text))) if text else 0
    except:
        return 0


def extract_engagement_stats(container):
    """Extract comment/retweet/like counts (optimized with single query)."""
    stats = {"retweet_count": 0, "like_count": 0, "comment_count": 0}

    try:
        # Get all stat elements at once
        icon_comment = container.find_elements(By.CSS_SELECTOR, ".icon-comment")
        icon_retweet = container.find_elements(By.CSS_SELECTOR, ".icon-retweet")
        icon_heart = container.find_elements(By.CSS_SELECTOR, ".icon-heart")
        
        if icon_comment:
            stats["comment_count"] = parse_engagement_number(
                icon_comment[0].find_element(By.XPATH, "..").text
            )
        if icon_retweet:
            stats["retweet_count"] = parse_engagement_number(
                icon_retweet[0].find_element(By.XPATH, "..").text
            )
        if icon_heart:
            stats["like_count"] = parse_engagement_number(
                icon_heart[0].find_element(By.XPATH, "..").text
            )
    except:
        pass

    return stats


def save_checkpoint(checkpoint_file, posts, seen_ids, scroll_count):
    """Save scraping progress (optimized with atomic write)."""
    checkpoint_data = {
        'posts': posts,
        'seen_ids': list(seen_ids),
        'scroll_count': scroll_count,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Write to temp file first, then rename (atomic operation)
    temp_file = f"{checkpoint_file}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    os.replace(temp_file, checkpoint_file)


# ---------------------------
# Scraper Core
# ---------------------------

def scrape_x_posts(username, num_scrolls=10, tweet_type="original", checkpoint_file=None):
    """Scrape tweets from Nitter with optimized performance."""
    
    if checkpoint_file is None:
        checkpoint_file = f"{username}_checkpoint.json"
    
    # Load existing progress
    posts = []
    seen_ids = set()
    start_scroll = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                posts = checkpoint_data.get('posts', [])
                seen_ids = set(checkpoint_data.get('seen_ids', []))
                start_scroll = checkpoint_data.get('scroll_count', 0)
                print(f"✓ Resuming from scroll {start_scroll} with {len(posts)} tweets already scraped")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}. Starting fresh...")

    # Optimized Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-images")  # Speed up by not loading images
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.page_load_strategy = 'eager'  # Don't wait for full page load
    
    # Additional performance preferences
    prefs = {
        "profile.managed_default_content_settings.images": 2,  # Disable images
        "profile.default_content_setting_values.notifications": 2,  # Disable notifications
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)
    wait = WebDriverWait(driver, 10)

    hashed_user = hash_username(username)
    
    # Pre-compile filter flags for tweet_type
    filter_retweets = tweet_type in ["original", "original_and_quotes"]
    filter_quotes = tweet_type == "original"

    try:
        driver.get(f"https://nitter.net/{username}")
        time.sleep(3)  # Reduced initial wait

        scrolls = start_scroll
        count = len(posts)
        last_height = driver.execute_script("return document.body.scrollHeight")
        no_new_content_count = 0
        
        while scrolls < num_scrolls:
            try:
                tweet_containers = wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".timeline-item"))
                )
            except TimeoutException:
                print(f"⚠ Timeout waiting for tweets at scroll {scrolls}")
                break

            new_tweets_this_scroll = 0

            for container in tweet_containers:
                try:
                    # Quick filters first
                    is_rt = is_retweet(container)
                    if filter_retweets and is_rt:
                        continue
                    
                    is_quote = is_quote_tweet(container)
                    if filter_quotes and is_quote:
                        continue

                    # Extract tweet ID early to check if seen
                    try:
                        date_element = container.find_element(By.CSS_SELECTOR, ".tweet-date a")
                        tweet_url = date_element.get_attribute("href")
                    except (NoSuchElementException, StaleElementReferenceException):
                        continue
                    
                    tweet_id = extract_tweet_id(tweet_url)
                    if not tweet_id or tweet_id in seen_ids:
                        continue
                    
                    seen_ids.add(tweet_id)

                    # Now extract all data
                    try:
                        tweet_element = container.find_element(By.CSS_SELECTOR, ".tweet-content")
                        tweet_text = tweet_element.text.strip().replace('"', "'")
                    except (NoSuchElementException, StaleElementReferenceException):
                        continue

                    parsed_date = parse_date(date_element.get_attribute("title"))
                    stats = extract_engagement_stats(container)

                    # Check if reply
                    is_reply = False
                    reply_to_id = None
                    try:
                        replying_to = container.find_element(By.CSS_SELECTOR, ".replying-to")
                        is_reply = True
                        reply_link = replying_to.find_element(By.TAG_NAME, "a").get_attribute("href")
                        reply_to_id = extract_tweet_id(reply_link)
                    except NoSuchElementException:
                        pass

                    # Extract additional data
                    urls = extract_urls(tweet_element)
                    hashtags = extract_hashtags(tweet_text)
                    mentions = extract_mentions(tweet_element)
                    media = extract_media(container)

                    posts.append({
                        "tweet_id": tweet_id,
                        "text": tweet_text,
                        "created_at": parsed_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "lang": detect_language(tweet_text),
                        "user_id_hashed": hashed_user,
                        "retweet_count": stats["retweet_count"],
                        "like_count": stats["like_count"],
                        "comment_count": stats["comment_count"],
                        "is_reply": is_reply,
                        "reply_to_id": reply_to_id,
                        "is_retweet": is_rt,
                        "is_quote": is_quote,
                        "urls": urls,
                        "hashtags": hashtags,
                        "mentions": mentions,
                        "media": media
                    })

                    count += 1
                    new_tweets_this_scroll += 1
                    print(f'{count} Tweets scraped', end='\r')

                except (StaleElementReferenceException, NoSuchElementException):
                    continue
                except Exception as e:
                    continue

            scrolls += 1
            
            # Save checkpoint every 5 scrolls instead of every scroll (reduces I/O)
            if scrolls % 5 == 0 or scrolls == num_scrolls:
                save_checkpoint(checkpoint_file, posts, seen_ids, scrolls)
                print(f'\n✓ Checkpoint saved: {len(posts)} tweets, scroll {scrolls}/{num_scrolls}')

            # Smart scrolling with early exit if no new content
            if new_tweets_this_scroll == 0:
                no_new_content_count += 1
                if no_new_content_count >= 3:
                    print(f"\n⚠ No new tweets found for 3 consecutive scrolls. Ending scrape.")
                    break
            else:
                no_new_content_count = 0

            # Optimized scrolling
            try:
                load_more = driver.find_element(By.XPATH, "//a[contains(text(),'Load more')]")
                driver.execute_script("arguments[0].click();", load_more)  # JS click is faster
                time.sleep(1.5)  # Reduced wait time
            except NoSuchElementException:
                # Check if we've reached the bottom
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    no_new_content_count += 1
                else:
                    last_height = new_height
                    
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)  # Reduced wait time

        # Final checkpoint save
        save_checkpoint(checkpoint_file, posts, seen_ids, scrolls)
        
        # Sort by date (only once at the end)
        posts.sort(key=lambda x: datetime.strptime(x["created_at"], "%Y-%m-%d %H:%M:%S"), reverse=True)

        # Clean up checkpoint on successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"\n✓ Checkpoint removed after successful completion")

        return posts

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Scraping interrupted by user!")
        print(f"Progress saved to {checkpoint_file}")
        print(f"Scraped {len(posts)} tweets before interruption")
        save_checkpoint(checkpoint_file, posts, seen_ids, scrolls)
        return posts  # Return what we have so far
        
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print(f"Progress saved to {checkpoint_file}")
        print(f"Scraped {len(posts)} tweets before error")
        save_checkpoint(checkpoint_file, posts, seen_ids, scrolls)
        return posts  # Return what we have so far

    finally:
        driver.quit()


# ---------------------------
# CSV Export
# ---------------------------

def export_to_csv(tweets, username, mode="timestamp", save_path=None):
    """Export tweets to CSV with optimized writing."""
    
    if not tweets:
        print("⚠ No tweets to export")
        return None
    
    # Create save directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    if mode == "timestamp":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_tweets_{timestamp}.csv"
        
    elif mode == "increment":
        base_filename = f"{username}_tweets"
        counter = 1
        filename = f"{base_filename}.csv"
        full_path = os.path.join(save_path, filename) if save_path else filename
        while os.path.exists(full_path):
            filename = f"{base_filename}_{counter}.csv"
            full_path = os.path.join(save_path, filename) if save_path else filename
            counter += 1
            
    elif mode == "ask":
        filename = f"{username}_tweets.csv"
        full_path = os.path.join(save_path, filename) if save_path else filename
        if os.path.exists(full_path):
            response = input(f"\n⚠️  File '{full_path}' already exists. Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{username}_tweets_{timestamp}.csv"
                print(f"Saving as: {filename}")
                
    elif mode == "merge":
        filename = f"{username}_tweets.csv"
        full_path = os.path.join(save_path, filename) if save_path else filename
        existing_tweets = []
        existing_ids = set()
        
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_tweets.append(row)
                        existing_ids.add(row['tweet_id'])
                print(f"✔ Found {len(existing_tweets)} existing tweets")
            except Exception as e:
                print(f"⚠ Could not load existing file: {e}")
        
        new_count = 0
        for tweet in tweets:
            if tweet['tweet_id'] not in existing_ids:
                existing_tweets.append(tweet)
                new_count += 1
        
        print(f"✔ Adding {new_count} new tweets")
        tweets = existing_tweets
        
        tweets.sort(
            key=lambda x: datetime.strptime(x["created_at"], "%Y-%m-%d %H:%M:%S"), 
            reverse=True
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_tweets_{timestamp}.csv"

    # Build full path
    full_path = os.path.join(save_path, filename) if save_path else filename

    fieldnames = [
        "tweet_id", "text", "created_at", "lang", "user_id_hashed",
        "retweet_count", "like_count", "comment_count",
        "is_reply", "reply_to_id", "is_retweet", "is_quote", 
        "urls", "hashtags", "mentions", "media"
    ]

    # Optimized CSV writing with buffering
    with open(full_path, "w", newline="", encoding="utf-8", buffering=8192) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in tweets:
            row = {
                "tweet_id": t["tweet_id"],
                "text": t["text"],
                "created_at": t["created_at"],
                "lang": t["lang"],
                "user_id_hashed": t["user_id_hashed"],
                "retweet_count": t["retweet_count"],
                "like_count": t["like_count"],
                "comment_count": t["comment_count"],
                "is_reply": t["is_reply"],
                "reply_to_id": t.get("reply_to_id"),
                "is_retweet": t["is_retweet"],
                "is_quote": t["is_quote"],
                "urls": json.dumps(t.get("urls", []), ensure_ascii=False),
                "hashtags": json.dumps(t.get("hashtags", []), ensure_ascii=False),
                "mentions": json.dumps(t.get("mentions", []), ensure_ascii=False),
                "media": json.dumps(t.get("media", []), ensure_ascii=False)
            }
            writer.writerow(row)

    return full_path


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    print("="*60)
    print("Enhanced Twitter Scraper - Optimized Edition")
    print("="*60)
    
    user = input("\nEnter username to scrape: ").strip()
    tweet_type = input("Choose tweet type ('original', 'original_and_quotes', 'all'): ").strip()

    if tweet_type not in ["original", "original_and_quotes", "all"]:
        print("⚠ Invalid type, defaulting to 'original'.")
        tweet_type = "original"

    print("\nCSV Save Mode Options:")
    print("  1. timestamp  - Add timestamp to filename (recommended)")
    print("  2. increment  - Add number suffix (_1, _2, etc.)")
    print("  3. merge      - Merge with existing CSV, avoid duplicates")
    print("  4. ask        - Ask before overwriting")
    
    save_mode = input("Choose save mode (default: timestamp): ").strip().lower()
    if save_mode not in ["timestamp", "increment", "merge", "ask"]:
        save_mode = "timestamp"

    # Define save path
    SAVE_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\data\raw"

    start_time = time.time()
    
    try:
        tweets = scrape_x_posts(user, num_scrolls=50, tweet_type=tweet_type)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✔ Successfully scraped {len(tweets)} tweets from @{user}")
        print(f"✔ Time taken: {elapsed_time:.2f} seconds ({len(tweets)/elapsed_time:.2f} tweets/sec)")
        print(f"{'='*60}\n")
        
        if tweets:
            # Display sample tweets
            for i, t in enumerate(tweets[:5], 1):
                kind = "RETWEET" if t["is_retweet"] else "QUOTE" if t["is_quote"] else "REPLY" if t["is_reply"] else "TWEET"
                print(f"{i}. [{kind}] [{t['created_at']}]")
                print(f"   {t['text'][:100]}{'...' if len(t['text']) > 100 else ''}")
                if t.get("hashtags"):
                    print(f"   Hashtags: {', '.join(t['hashtags'][:3])}")
                if t.get("mentions"):
                    print(f"   Mentions: {', '.join(t['mentions'][:3])}")
                print()
            
            if len(tweets) > 5:
                print(f"... and {len(tweets) - 5} more tweets\n")

            filename = export_to_csv(tweets, user, mode=save_mode, save_path=SAVE_PATH)
            if filename:
                print(f"✔ Saved to {filename}")
        
    except KeyboardInterrupt:
        print("\n\nScraping interrupted. Partial data may be saved in checkpoint file.")
        
    except Exception as e:
        print(f"\n\nâŒ An error occurred: {e}")
        print("Check the checkpoint file to resume scraping later.")