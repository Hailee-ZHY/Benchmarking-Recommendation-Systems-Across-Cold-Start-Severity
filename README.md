# Benchmarking-Recommendation-Systems-Across-Cold-Start-Severity

##  Review Dataset Overview

* **Total Rows**: 29,475,453 reviews
* **Original Size**: ~5.8 GB (Compressed)
* **Storage Format**: Loaded via Hugging Face `datasets` and persisted as local **Parquet** for optimized Spark performance.

## Schema Definition (https://amazon-reviews-2023.github.io/)
| Column | Type | Description |
| :--- | :--- | :--- |
| **rating** | double | Star rating (1.0 to 5.0) |
| **title** | string | Review headline |
| **text** | string | Full body of the review |
| **images** | array | Metadata for images attached to reviews |
| **asin** | string | ID of the product |
| **parent_asin** | string | Parent ID of the product. Note: Products with different colors, styles, sizes usually belong to the same parent ID. The “asin” in previous Amazon datasets is actually parent ID. Please use parent ID to find product meta. |
| **user_id** | string | Unique reviewer ID |
| **timestamp** | bigint | Unix timestamp of the review |
| **helpful_vote** | bigint | Number of "Helpful" votes received |
| **verified_purchase** | boolean | Whether the purchase was confirmed by Amazon |

## Title & Text & images examples: Top 5
 I also tried to introduce images here, but found they are useless. e.g. https://m.media-amazon.com/images/I/516HBU7LQoL._SL1600_.jpg
| Title | Text |
| :--- | :--- |
| Beautiful patterns! | I love this book!  The patterns are lovely. I admit I’m not even following all of them to the letter. I am more of a free spirit.  I see the designs & take a look at the dimensions & just kind of wing it from there sometimes. lol. I’m 54 now & much less rigid in my arts & crafts than I was in my younger days when I would go to 6 stores trying to find the right threads or yarns. Lol. Those days are over!  Praise the Lord!  It’s a lot more fun now too.  As Bob Ross used to say- “Happy Mistakes!”  I make a lot of them sometimes, but I have found that the journey for me is a lot more interesting these days. Lol. |
| Excellent! I love it!  | I bought it for the bag on the front so it paid for itself with that imo.  I haven’t started anything yet from it bc I’m still busy with other projects, but I’m really looking forward to starting some. |
| Not a watercolor book! Seems like copies imo. | It is definitely not a watercolor book.  The paper bucked completely.  The pages honestly appear to be photo copies of other pictures. I say that bc if you look at the seal pics you can see the tell tale line at the bottom of the page.  As someone who has made many photocopies of pages in my time so I could try out different colors & mediums that black line is a dead giveaway to me. It’s on other pages too.  The entire book just seems off. Nothing is sharp & clear. There is what looks like toner dust on all the pages making them look muddy.  There are no sharp lines & there is no clear definition.  At least there isn’t in my copy.  And the Coloring Book for Adult on the bottom of the front cover annoys me. Why is it singular & not plural?  They usually say coloring book for kids or coloring book for kids & adults or coloring book for adults- plural.  Lol  Plus it would work for kids if you can get over the grey scale nature of it.  Personally I’m not going to waste expensive pens & paints trying to paint over the grey & black mess.  I grew up in SW Florida minutes from the beaches & I was really excited about the sea life in this. I hope the printers & designers figure out how to clean up the mess bc some of the designs are really cute. They just aren’t worth my time to hand trace & transfer them, but I’m sure there are ppl that will be up to the challenge.  This is one is a hard no. Going back. I tried. |

##  Meta Dataset Overview
* **Total Rows**: 4,448,181 products
* **Original Size**: ~GB (Compressed)

## Schema Definition (need to be updated)
| Column | Type | Description |
| :--- | :--- | :--- |
| main_category   | string     | Product domain/category |
| title           | string     | Product title |
| average_rating  | double     | Average user rating |
| rating_number   | bigint     | Number of ratings |
| features        | array      | Product bullet features |
| description     | array      | Product description |
| price           | string     | Price (string format) |
| images          | struct     | Image URLs (hi_res, large, thumb, variant) |
| videos          | struct     | Video info (title, url, user_id) |
| store           | string     | Store name |
| categories      | array      | Hierarchical categories |
| details         | string     | Additional product details |
| parent_asin     | string     | Parent product ID |
| bought_together | string     | Frequently bought together |
| subtitle        | string     | Subtitle |
| author          | string     | Author |

## Environment & Configuration
* **Engine**: PySpark (Local Mode)
* **Source**: `McAuley-Lab/Amazon-Reviews-2023` (raw_review_Books)
* **Optimization**: Adjusted `spark.driver.memory` (4g) and `spark.sql.shuffle.partitions` (20) to prevent `OutOfMemoryError` on Intel-based hardware.
