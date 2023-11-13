float cx, cy, rad; 
float shift = 0;
int shift_every = 1;
int update_every = 18;
boolean auto_play = true;
boolean is_saving = false;

W w = new W();

float adj(float cur, float tar) { float d = tar - cur, ad = abs(d); if (ad < 0.01) return tar; else return cur + 0.4 * d / sqrt(sqrt(ad)); }

class W {
   C[] cs;
   W() {
      cs = new C[0]; 
   }
   
   void add_string(String s) { 
     for (int i = 0; i < s.length(); i++) { add_c(new C(i, s.charAt(i))); }
   }
      
   void cyclic_cut_insert(int loc, int cut_len, String s) { // s: new token
     for (int i = 0; i < min(cut_len, s.length()); i++) { int j = (loc + i) % cs.length; cs[j].c = s.charAt(i); } // the good common part
     for (int i = min(cut_len, s.length()); i < cut_len; i++) { int j = (loc + i) % cs.length; cs[j] = null; } // the garbage when cut_len > s.length
     for (int i = min(cut_len, s.length()); i < s.length(); i++) { int j = (loc + i) % cs.length; add_c(new C(j, s.charAt(i)), j); }
     C[] old_cs = cs; 
     // cleanup
     cs = new C[0];
     for (int i = 0; i < old_cs.length; i++) {
        if (old_cs[i] == null) continue;
        cs = (C[]) append(cs, old_cs[i]);
        cs[cs.length - 1].rank = cs.length - 1;
     }
   }
   
   void cyclic_cut_insert(E e) { cyclic_cut_insert(e.loc, e.cut_len, e.s); }
   
   void add_c(C c) { cs = (C[]) append(cs, c); }
   void add_c(C c, int i) { cs = (C[]) concat(append(subset(cs, 0, i), c), subset(cs, i));  }
   void remove_c(int ind) { cs = (C[]) concat(subset(cs, 0, ind), subset(cs, ind + 1)); }
   
   
   void draw() { for (C c : cs) c.draw(); }
   
   void evo() { for (C c : cs) c.evo(); }
}

class C {
   int rank;
   char c;
   float x, y, ang;
   float sx, sy, sang; // the soft versions of the angles
   
   C(int _rank, char _c) { rank = _rank; c = _c; sx = width / 2; sy = height / 2; sang = 0; evo();  }
   
   void draw() {
        pushMatrix();
        translate(sx, sy); rotate(sang + HALF_PI);
        text(c, 0, 0);
        popMatrix();
   }
   
   void evo() {
       ang = TWO_PI * (float(rank) + shift) / (w.cs.length + 1) - HALF_PI;
       x = cx + rad * cos(ang); y = cy + rad * sin(ang); 
       sx = adj(sx, x); sy = adj(sy, y); sang = adj(sang, ang);
   }
}

class E {
   int loc, cut_len; String s;
   E(int _loc, int _cut_len, String _s) { loc = _loc; cut_len = _cut_len; s = _s; }
}

E[] es = new E[0];
int ie = -1;
void add_new_e(int loc, int cut_len, String s) { es = (E[]) append(es, new E(loc, cut_len, s)); }

void setup() {
  size(800, 800);
  cx = width / 2; cy = height / 2; rad = 340;
  // w.add_string("Hello, it's a world of wonders and a universe of magic");
  loadJson();
  textFont(loadFont("CourierNewPSMT-36.vlw"));
}

void loadJson() {
   JSONObject json = loadJSONObject("circle-dynamics.json");
   if (json == null) { println("Missing json file"); return; }
   String initString = json.getString("init_string", "");
   if (initString.length() == 0) { println("Missing init string"); return; }
   w.add_string(initString);
   JSONArray editArray = json.getJSONArray("edits");
   if (editArray.size() == 0) { println("Missing edits"); return; }
   for (int i = 0; i < editArray.size(); i++) {
     JSONObject editJson = editArray.getJSONObject(i);
     int loc = editJson.getInt("loc", 0);
     int cut_len = editJson.getInt("cut_len", 0);
     String s = editJson.getString("token", "");
     add_new_e(loc, cut_len, s);
   }
}

void play() {
      ie++; if (ie < es.length) { w.cyclic_cut_insert(es[ie]); }
}



void draw() {
  background(0);
  w.draw();
  w.evo();
  if (auto_play && (frameCount + 1) % update_every == 0) play();
  if (auto_play && (frameCount + 1) % shift_every == 0) { shift -= 0.03; }
  if (is_saving) {
     saveFrame(hour() + "_" + minute()+ "####.png"); 
  }
//  if ((frameCount + 1) % update_every == 0) { ie++; if (ie < es.length) { w.cyclic_cut_insert(es[ie]); } }
}

void keyPressed(KeyEvent keyEvent) {
  if (keyCode == 32) {
    play();
  }
  if (keyCode == LEFT) {
     shift--;
  }
  if (keyCode == RIGHT) {
    shift++;
  }
  if (keyCode == 'S' && (keyEvent.isControlDown() || keyEvent.isMetaDown())) {
    is_saving ^= true;
  }
}
