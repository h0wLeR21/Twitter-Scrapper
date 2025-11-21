# Data Dictionary

## Tweet Data Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| tweet_id | string | Unique tweet identifier | "1234567890" |
| text | string | Tweet content | "Hello world!" |
| created_at | datetime | Publication timestamp | "2024-01-15 10:30:00" |
| lang | string | ISO language code | "en" |
| user_id_hashed | string | Anonymized user ID | "a1b2c3d4..." |
| retweet_count | integer | Number of retweets | 42 |
| like_count | integer | Number of likes | 156 |
| comment_count | integer | Number of replies | 23 |
| is_reply | boolean | Is this a reply? | true/false |
| reply_to_id | string | Parent tweet ID | "9876543210" |
| is_retweet | boolean | Is this a retweet? | true/false |
| is_quote | boolean | Is this a quote tweet? | true/false |
| urls | json array | Extracted URLs | ["https://..."] |
| hashtags | json array | Extracted hashtags | ["#python"] |
| mentions | json array | Extracted mentions | ["@user"] |
| media | json array | Media attachments | [{type, url}] |

## Media Object Structure

```json
{
  "type": "image|video",
  "url": "https://...",
  "alt": "Alt text"
}
```

## Data Privacy

- User IDs are hashed using SHA-256
- Only publicly available data is collected
- No authentication tokens are stored
