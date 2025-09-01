import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

export const FetchData = express.Router();

const PAGE_ID = process.env.PAGE_ID;
const PAGE_ACCESS_TOKEN = process.env.PAGE_ACCESS_TOKEN;

// 🟢 Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// 🔄 Function to fetch & save posts
async function fetchFacebookPosts(limit = 10) {
  try {
    console.log("📡 Fetching latest posts from Facebook...");

    // 1️⃣ Fetch page info
    const pageUrl = `https://graph.facebook.com/v18.0/${PAGE_ID}?fields=name,picture.type(large)&access_token=${PAGE_ACCESS_TOKEN}`;
    const pageResp = await fetch(pageUrl);
    const pageData = await pageResp.json();
    if (pageData.error) throw new Error(pageData.error.message);

    const pageName = pageData.name;
    const pageLogo = pageData.picture?.data?.url || null;

    // 2️⃣ Fetch posts
    const postsUrl = `https://graph.facebook.com/v18.0/${PAGE_ID}/feed?fields=id,message,permalink_url,created_time,attachments{media,subattachments}&limit=${limit}&access_token=${PAGE_ACCESS_TOKEN}`;
    const postsResp = await fetch(postsUrl);
    const postsData = await postsResp.json();
    if (postsData.error) throw new Error(postsData.error.message);

    console.log(`✅ ${postsData.data?.length || 0} posts fetched`);

    // 3️⃣ Normalize posts
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
        fb_post_id: post.id,              // required column
        permalink: post.permalink_url,    // unique identifier
        content: post.message || null,
        images,                           // array column
        created_at: new Date(post.created_time),
        page_name: pageName,
        logo_url: pageLogo,
        last_synced: new Date().toISOString(),
      };
    });

    // 4️⃣ Upsert posts using permalink as conflict key (no duplicates)
    for (const post of posts) {
      const { error, status } = await supabase
        .from("posts")
        .upsert(post, {
          onConflict: "permalink",
          ignoreDuplicates: true, // ✅ ensures no reinsert of existing posts
        });

      if (error) {
        console.error("❌ Supabase insert error:", error.message);
      } else if (status === 409) {
        console.log(`⚠️ Skipped duplicate post: ${post.permalink}`);
      } else {
        console.log(`✅ Saved/updated post: ${post.permalink}`);
      }
    }

    return {
      page: { id: PAGE_ID, name: pageName, logo: pageLogo },
      posts,
      paging: postsData.paging?.cursors || null,
    };

  } catch (err) {
    console.error("💥 Error fetching Facebook posts:", err.message);
    return { error: err.message };
  }
}

// 🟢 API route for manual trigger
FetchData.get("/api/posts", async (req, res) => {
  const result = await fetchFacebookPosts(Number(req.query.limit) || 10);
  if (result.error) return res.status(500).json({ error: result.error });
  res.json(result);
});

// ⏱ Auto-fetch every 1 minute (background job)
setInterval(() => {
  fetchFacebookPosts(10);
}, 60 * 1000);
