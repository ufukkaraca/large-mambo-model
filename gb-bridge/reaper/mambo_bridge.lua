--[[ Mambo → REAPER bridge (the actuation layer, PAPER §4.8).

Load this once in REAPER (Actions → Load ReaScript → Run). It watches the inbox
folder for `*.plan.json` files that the Mambo pipeline drops (a mambo.action.v1
plan + a rendered .mid) and applies each: inserts the hummed notes, swaps the
instrument ("make it electric"), loops, nudges faders, etc. Leave it running.

Setup: edit CONFIG below if your repo path differs, then map PATCHES to your own
instrument VSTs for real sounds (defaults to stock ReaSynth so it works out of
the box). Stop it via Actions → (running scripts) → Terminate.
--]]

------------------------------------------------------------ CONFIG
local INBOX = "/Users/ufuk_/Developer/tinkers/large-mambo-model/out/reaper_inbox"
local DONE  = INBOX .. "/done"
local MAMBO_TRACK = "Mambo"
local POLL_SEC = 0.4
-- patch name -> instrument VST (use your own for real sounds; ReaSynth is stock)
local PATCHES = {
  default        = "ReaSynth",
  piano          = "ReaSynth", bright_piano = "ReaSynth",
  electric_piano = "ReaSynth", strings = "ReaSynth",
  synth          = "ReaSynth", warm_pad = "ReaSynth",
}
------------------------------------------------------------ JSON (minimal decode, MIT rxi/json.lua)
local json = {} do
  local esc={['"']='"',['\\']='\\',['/']='/',b='\b',f='\f',n='\n',r='\r',t='\t'}
  local function skip(s,i) while i<=#s and s:find("^[ \t\r\n]",i) do i=i+1 end return i end
  local pv
  local function pstr(s,i) local r="" i=i+1 while i<=#s do local c=s:sub(i,i)
      if c=='"' then return r,i+1 elseif c=='\\' then local n=s:sub(i+1,i+1)
        if n=='u' then r=r.."?" i=i+6 else r=r..(esc[n] or n) i=i+2 end
      else r=r..c i=i+1 end end error("bad string") end
  local function pnum(s,i) local j=i while i<=#s and s:find("^[%d%.eE%+%-]",i) do i=i+1 end
      return tonumber(s:sub(j,i-1)),i end
  local function parr(s,i) local r={} i=skip(s,i+1) if s:sub(i,i)==']' then return r,i+1 end
      while true do local v v,i=pv(s,i) r[#r+1]=v i=skip(s,i) local c=s:sub(i,i)
        if c==']' then return r,i+1 end i=skip(s,i+1) end end
  local function pobj(s,i) local r={} i=skip(s,i+1) if s:sub(i,i)=='}' then return r,i+1 end
      while true do local k k,i=pstr(s,i) i=skip(s,i)+1 i=skip(s,i) local v v,i=pv(s,i)
        r[k]=v i=skip(s,i) local c=s:sub(i,i) if c=='}' then return r,i+1 end i=skip(s,i+1) end end
  pv=function(s,i) i=skip(s,i) local c=s:sub(i,i)
      if c=='{' then return pobj(s,i) elseif c=='[' then return parr(s,i)
      elseif c=='"' then return pstr(s,i)
      elseif c=='t' then return true,i+4 elseif c=='f' then return false,i+5
      elseif c=='n' then return nil,i+4 else return pnum(s,i) end end
  function json.decode(s) local ok,v=pcall(function() return (pv(s,1)) end)
      if ok then return v end return nil end
end
------------------------------------------------------------ helpers
local function log(m) reaper.ShowConsoleMsg("[mambo] "..m.."\n") end

local function read_file(p) local f=io.open(p,"r") if not f then return nil end
  local c=f:read("*a") f:close() return c end

local function find_track(name)
  for i=0,reaper.CountTracks(0)-1 do local t=reaper.GetTrack(0,i)
    local _,n=reaper.GetSetMediaTrackInfo_String(t,"P_NAME","",false)
    if n==name then return t end end
  return nil
end

local function find_or_create(name)
  local t=find_track(name)
  if not t then reaper.InsertTrackAtIndex(reaper.CountTracks(0),true)
    t=reaper.GetTrack(0,reaper.CountTracks(0)-1)
    reaper.GetSetMediaTrackInfo_String(t,"P_NAME",name,true) end
  return t
end

local function mambo_track() return find_or_create(MAMBO_TRACK) end

-- Resolve THIS action's target track: by name (find-or-create, robust to track
-- order), else by session index, else the Mambo track. So "kick the drums up"
-- hits a "Drums" track and "make the bass louder" hits "Bass".
local function resolve_track(args, fallback)
  local tr = args.track or args   -- volume/instrument nest under .track; mute/solo carry by/index directly
  if type(tr)=="table" then
    if tr.by and tr.by ~= "selected" then return find_or_create(tr.by) end
    if tr.index ~= nil then local t=reaper.GetTrack(0, math.floor(tr.index)); if t then return t end end
  end
  return fallback
end

local function ensure_instrument(t, patch)
  local name = PATCHES[patch or "default"] or PATCHES.default
  -- if the track has no instrument FX, add the configured one
  if reaper.TrackFX_GetInstrument(t) < 0 then
    reaper.TrackFX_AddByName(t, name, false, 1)
  end
end
------------------------------------------------------------ op handlers
local function op_insert(t, a)
  reaper.SetOnlyTrackSelected(t)
  if a.tempo_bpm then reaper.SetCurrentBPM(0, a.tempo_bpm, true) end
  ensure_instrument(t, nil)
  reaper.SetEditCurPos(0, false, false)
  local mf = a._midi_file
  if mf and read_file(mf) then reaper.InsertMedia(mf, 0); log("inserted "..mf) else log("no MIDI to insert") end
end

local function op_instrument(t, a)
  local name = PATCHES[a.patch or "default"] or PATCHES.default
  local idx = reaper.TrackFX_GetInstrument(t)
  if idx >= 0 then reaper.TrackFX_Delete(t, idx) end
  reaper.TrackFX_AddByName(t, name, false, 1)
  -- crude timbre differentiation on stock ReaSynth (best-effort; pcall-safe)
  local wave = ({piano=0.0, bright_piano=0.25, electric_piano=0.5, synth=0.75, warm_pad=0.9})[a.patch] or 0.0
  pcall(function() reaper.TrackFX_SetParamNormalized(t, reaper.TrackFX_GetInstrument(t), 1, wave) end)
  log("instrument -> "..(a.patch or "?"))
end

local function op_volume(t, a)
  local v = reaper.GetMediaTrackInfo_Value(t,"D_VOL")
  reaper.SetMediaTrackInfo_Value(t,"D_VOL", v * 10^((a.delta_db or 0)/20))
  log("volume "..tostring(a.delta_db).." dB")
end

local function track_extent(t)
  local n=reaper.CountTrackMediaItems(t); local lo,hi=0,4
  for i=0,n-1 do local it=reaper.GetTrackMediaItem(t,i)
    local p=reaper.GetMediaItemInfo_Value(it,"D_POSITION")
    local l=reaper.GetMediaItemInfo_Value(it,"D_LENGTH")
    if p+l>hi then hi=p+l end end
  return lo,hi
end

local function op_transport(t, a)
  local act=a.action
  if act=="play" then reaper.CSurf_OnPlay()
  elseif act=="stop" then reaper.CSurf_OnStop()
  elseif act=="record" then reaper.CSurf_OnRecord()
  elseif act=="to_start" then reaper.SetEditCurPos(0,true,true)
  elseif act=="cycle" then
    local lo,hi=track_extent(t)
    reaper.GetSet_LoopTimeRange(true,true,lo,hi,false)
    reaper.GetSetRepeat(1); reaper.SetEditCurPos(lo,true,true); reaper.CSurf_OnPlay()
  end
  log("transport "..tostring(act))
end

local function apply(plan)
  local t=mambo_track()
  -- collect midi artifact per insert op
  local midi=nil
  for _,a in ipairs(plan.actions or {}) do
    if a.artifacts and a.artifacts.midi_file then midi=a.artifacts.midi_file end
  end
  reaper.Undo_BeginBlock()
  for _,a in ipairs(plan.actions or {}) do
    local args=a.args or {}
    local tt = resolve_track(args, t)   -- per-action target ("Drums", "Bass", … or the Mambo track)
    if a.op=="insert_notes" then args._midi_file=midi; pcall(op_insert,tt,args)
    elseif a.op=="set_track_instrument" then pcall(op_instrument,tt,args)
    elseif a.op=="change_track_volume" then pcall(op_volume,tt,args)
    elseif a.op=="mute_track" then reaper.SetMediaTrackInfo_Value(tt,"B_MUTE",1)
    elseif a.op=="solo_track" then reaper.SetMediaTrackInfo_Value(tt,"I_SOLO",1)
    elseif a.op=="pan_track" then reaper.SetMediaTrackInfo_Value(tt,"D_PAN", math.max(-1,math.min(1, reaper.GetMediaTrackInfo_Value(tt,"D_PAN")+(args.delta_pan or 0))))
    elseif a.op=="undo" then reaper.Undo_DoUndo2(0)
    elseif a.op=="record_take" then reaper.SetOnlyTrackSelected(tt); reaper.SetMediaTrackInfo_Value(tt,"I_RECARM",1); reaper.CSurf_OnRecord()
    elseif a.op=="set_project_tempo" then reaper.SetCurrentBPM(0,args.bpm or 120,true)
    elseif a.op=="transport" then pcall(op_transport,t,args)
    elseif a.op=="play_preview" then -- preview is the co-pilot's own synth; no-op in REAPER
    elseif a.op=="ask_user" then log("ASK: "..tostring(args.question)) end
  end
  reaper.Undo_EndBlock("Mambo: "..(plan.intent_summary or "plan"), -1)
  reaper.UpdateArrange()
end
------------------------------------------------------------ watch loop
os.execute('mkdir -p "'..DONE..'"')
local seen={}
local last_hb=0
local function heartbeat()  -- a fresh timestamp file so Studio/CLI can see REAPER is listening
  local now=os.time()
  if now-last_hb>=1 then last_hb=now
    local f=io.open(INBOX.."/.heartbeat","w"); if f then f:write(tostring(now)); f:close() end end
end
local function tick()
  heartbeat()
  local i=0
  while true do
    local f=reaper.EnumerateFiles(INBOX,i); i=i+1
    if not f then break end
    if f:match("%.plan%.json$") and not seen[f] then
      seen[f]=true
      local txt=read_file(INBOX.."/"..f)
      local plan=txt and json.decode(txt)
      if plan then log("apply "..f.." :: "..(plan.intent_summary or "")); apply(plan)
        os.rename(INBOX.."/"..f, DONE.."/"..f) end
    end
  end
  reaper.defer(tick)
end
log("watching "..INBOX)
tick()
