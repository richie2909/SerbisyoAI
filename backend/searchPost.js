import express from "express";
import fetch from "node-fetch";

const router = express.Router();

const PAGE_ID = process.env.PAGE_ID;
const PAGE_ACCESS_TOKEN = process.env.PAGE_ACCESS_TOKEN;

// Helper to fetch all posts from Facebook
async function fetchAllPosts(limit = 50) {
  const postsUrl = `https://graph.facebook.com/v18.0/${PAGE_ID}/feed?fields=message,permalink_url,created_time,attachments{media,subattachments}&limit=${limit}&access_token=${PAGE_ACCESS_TOKEN}`;
  const postsResp = await fetch(postsUrl);
  const postsData = await postsResp.json();

  if (postsData.error) throw new Error(postsData.error.message);

  const posts = (postsData.data || []).map(post => {
    const images = [];

    if (post.attachments?.data) {
      post.attachments.data.forEach(att => {
        if (att.subattachments?.data) {
          att.subattachments.data.forEach(sub => {
            if (sub.media?.image?.src) images.push(sub.media.image.src);
          });
        } else if (att.media?.image?.src) {
          images.push(att.media.image.src);
        }
      });
    }

    return {
      caption: post.message || "",
      images,
      permalink: post.permalink_url,
      timestamp: post.created_time,
    };
  });

  return posts;
}

router.get("/api/search-posts", async (req, res) => {
  try {
    const query = (req.query.q)?.toLowerCase() || "";
    const limit = Number(req.query.limit) || 20;

    if (!query) return res.json({ posts: [] });

    // Fetch all posts (or limit initial fetch)
    const postsUrl = `https://graph.facebook.com/v18.0/${PAGE_ID}/feed?fields=message,permalink_url,created_time,attachments{media,subattachments}&limit=100&access_token=${PAGE_ACCESS_TOKEN}`;
    const postsResp = await fetch(postsUrl);
    const postsData = await postsResp.json();

    const posts = (postsData.data || []).map(post => {
      const images = [];

      if (post.attachments?.data) {
        post.attachments.data.forEach(att => {
          if (att.subattachments?.data) {
            att.subattachments.data.forEach(sub => {
              if (sub.media?.image?.src) images.push(sub.media.image.src);
            });
          } else if (att.media?.image?.src) images.push(att.media.image.src);
        });
      }

      return {
        caption: post.message || "",
        images,
        permalink: post.permalink_url,
        timestamp: post.created_time,
        page_name: "", // optionally attach page name
      };
    });

    // Filter by caption text or hashtags
    const filtered = posts.filter(p => {
      const lowerCaption = p.caption.toLowerCase();
      return lowerCaption.includes(query) || lowerCaption.split(" ").some(w => w.startsWith("#") && w.includes(query));
    });

    res.json({ posts: filtered.slice(0, limit) });

  } catch (err) {
    console.error("Search error:", err);
    res.status(500).json({ error: err.message });
  }
});


// ------------------------
// Search endpoint
// ------------------------


export default router;
