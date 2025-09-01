async function syncPostsToSupabase(posts) {
  if (!posts || posts.length === 0) {
    console.log("ℹ️ No posts to sync.");
    return;
  }

  const formattedPosts = posts.map(post => {
    // Collect multiple images if attachments exist
    let images = [];
    if (post.attachments?.data) {
      images = post.attachments.data
        .map(att => att.media?.image?.src)
        .filter(Boolean);
    } else if (post.full_picture) {
      images = [post.full_picture];
    }

    return {
      fb_post_id: post.id,                            // ✅ required
      permalink: post.permalink_url || null,          // ✅ permalink
      caption: post.message || null,                  // ✅ caption
      images,                                         // ✅ array of images
      page_name: post.page_name || null,              // optional
      logo_url: post.logo_url || null,                // optional
      last_synced: new Date().toISOString(),          // ✅ timestamp
    };
  });

  const { data, error } = await supabase
    .from("posts")
    .upsert(formattedPosts, { onConflict: "fb_post_id" });

  if (error) {
    console.error("❌ Supabase insert error:", error.message, error.details);
  } else {
    console.log(`✅ Synced ${data?.length || 0} posts`);
  }
}
