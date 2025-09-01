// // import Express from "express";
// // import fetch from "node-fetch"; // If Node 18+, you can skip this
// // import { createClient } from "@supabase/supabase-js";

// // const Router = Express.Router();

// // // Replace these with your actual Supabase URL and service role key
// // const SUPABASE_URL = process.env.SUPABASE_URL;
// // const SUPABASE_SERVICE_KEY= process.env.SUPABASE_SERVICE_KEY;

// // const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// // // Facebook Page credentials
// // const PAGE_ID = process.env.PAGE_ID;
// // const PAGE_ACCESS_TOKEN = process.env.PAGE_ACCESS_TOKEN;

// // // ✅ Route 1: Test Facebook token
// // Router.get("/api/test-token", async (req, res) => {
// //   try {
// //     const url = `https://graph.facebook.com/v18.0/${PAGE_ID}?fields=id,name&access_token=${PAGE_ACCESS_TOKEN}`;
// //     const response = await fetch(url);
// //     const data = await response.json();

// //     if (data.error) throw new Error(data.error.message);

// //     res.json({ success: true, page: data });
// //   } catch (err) {
// //     res.json({ success: false, error: err.message });
// //   }
// // });

// // // ✅ Route 2: Invite Supabase user
// // Router.post("/api/invite-user", async (req, res) => {
// //   const { email, full_name } = req.body;

// //   if (!email) return res.status(400).json({ success: false, error: "Email is required" });

// //   try {
// //     // Create Supabase user
// //     const { data: user, error } = await supabase.auth.admin.createUser({
// //       email,
// //       email_confirm: true,
// //       user_metadata: { full_name: full_name || null }
//     });

//     if (error) throw error;

//     // Manually insert profile
//     const { error: profileError } = await supabase
//       .from("profiles")
//       .insert([{ user_id: user.id, full_name: full_name || null }])
//       .select();

//     if (profileError) throw profileError;

//     res.json({ success: true, user });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ success: false, error: err.message });
//   }
// });

// export default Router;
