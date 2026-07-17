-- pranava · Śabda-ALM — Supabase schema
-- Access model (operator's design):
--   • admin  → live DGX-Spark/5090 inference via the cloudflare tunnel; may invite guests.
--   • guest  → live inference (email allow-listed by an admin).
--   • user   → default; REPLAY of stored demo artifacts only (no GPU access).
-- Mirrors the prabhasa-samskrutam / prabodha app pattern (user_tiers + runtime_config + RLS).
-- Run in the Supabase SQL editor after project creation.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- Account tiers
-- ---------------------------------------------------------------------------
create type public.account_tier as enum ('user', 'guest', 'admin');

create table if not exists public.user_tiers (
  user_id     uuid primary key references auth.users (id) on delete cascade,
  email       text,
  tier        public.account_tier not null default 'user',
  created_at  timestamptz not null default now()
);
alter table public.user_tiers enable row level security;

drop policy if exists "user_tiers select self" on public.user_tiers;
create policy "user_tiers select self" on public.user_tiers
  for select using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- Live-access allow-list (admin-managed). An email here → 'guest' on signup.
-- ---------------------------------------------------------------------------
create table if not exists public.live_allowlist (
  email       text primary key,
  invited_by  uuid references auth.users (id),
  created_at  timestamptz not null default now()
);
alter table public.live_allowlist enable row level security;

drop policy if exists "allowlist admin read" on public.live_allowlist;
create policy "allowlist admin read" on public.live_allowlist
  for select using (exists (select 1 from public.user_tiers t
                            where t.user_id = auth.uid() and t.tier = 'admin'));

-- The admin bootstrap email (operator). Seeded so the first signup with this address is admin.
insert into public.live_allowlist (email) values ('sharath.sathish@gmail.com')
  on conflict (email) do nothing;

-- ---------------------------------------------------------------------------
-- Assign a tier on signup: admin email → admin; allow-listed → guest; else user.
-- ---------------------------------------------------------------------------
create or replace function public.assign_tier() returns trigger
language plpgsql security definer set search_path = public as $$
declare t public.account_tier;
begin
  if new.email = 'sharath.sathish@gmail.com' then
    t := 'admin';
  elsif exists (select 1 from public.live_allowlist a where a.email = new.email) then
    t := 'guest';
  else
    t := 'user';
  end if;
  insert into public.user_tiers (user_id, email, tier) values (new.id, new.email, t)
    on conflict (user_id) do update set tier = excluded.tier, email = excluded.email;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users for each row execute function public.assign_tier();

-- Admin RPC: invite a guest by email (adds to allow-list; upgrades them if already signed up).
create or replace function public.invite_guest(guest_email text) returns json
language plpgsql security definer set search_path = public as $$
begin
  if not exists (select 1 from public.user_tiers where user_id = auth.uid() and tier = 'admin') then
    raise exception 'admin only';
  end if;
  insert into public.live_allowlist (email, invited_by) values (lower(guest_email), auth.uid())
    on conflict (email) do nothing;
  update public.user_tiers set tier = 'guest'
    where email = lower(guest_email) and tier = 'user';
  return json_build_object('invited', lower(guest_email));
end $$;

-- ---------------------------------------------------------------------------
-- Runtime config — the live gateway (cloudflare tunnel) URL. Public-read so the
-- app can find the backend; admin-write only (via RPC).
-- ---------------------------------------------------------------------------
create table if not exists public.runtime_config (
  key         text primary key,
  value       text,
  updated_at  timestamptz not null default now()
);
alter table public.runtime_config enable row level security;

drop policy if exists "config public read" on public.runtime_config;
create policy "config public read" on public.runtime_config for select using (true);

insert into public.runtime_config (key, value) values
  ('alm_gateway_url', ''),                      -- set by the tunnel service on the DGX
  ('model', 'Śabda-ALM (200M, instruction-tuned)')
  on conflict (key) do nothing;

create or replace function public.set_runtime_config(key_name text, key_value text) returns json
language plpgsql security definer set search_path = public as $$
begin
  if not exists (select 1 from public.user_tiers where user_id = auth.uid() and tier = 'admin') then
    raise exception 'admin only';
  end if;
  insert into public.runtime_config (key, value, updated_at) values (key_name, key_value, now())
    on conflict (key) do update set value = excluded.value, updated_at = now();
  return json_build_object('key', key_name, 'value', key_value);
end $$;

-- ---------------------------------------------------------------------------
-- Demo artifacts — precomputed (audio-in → spoken answer) samples the REPLAY
-- experience serves to 'user' accounts (no GPU). Public-read to any signed-in user.
-- ---------------------------------------------------------------------------
create table if not exists public.demo_artifacts (
  id          uuid primary key default gen_random_uuid(),
  clip_id     text not null,
  task        text not null,
  gold        text,
  answer      text not null,           -- the model's text answer
  audio_in    text,                    -- storage path / public URL of the input clip
  audio_out   text,                    -- storage path / public URL of the spoken answer
  lang        text,
  created_at  timestamptz not null default now()
);
alter table public.demo_artifacts enable row level security;

drop policy if exists "demo read authed" on public.demo_artifacts;
create policy "demo read authed" on public.demo_artifacts
  for select using (auth.role() = 'authenticated');

-- ---------------------------------------------------------------------------
-- Speak history — per-user log of live interactions (owner-only).
-- ---------------------------------------------------------------------------
create table if not exists public.speak_history (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade,
  task        text not null,
  answer      text,
  lang        text,
  created_at  timestamptz not null default now()
);
alter table public.speak_history enable row level security;

drop policy if exists "history own read" on public.speak_history;
create policy "history own read" on public.speak_history
  for select using (auth.uid() = user_id);
drop policy if exists "history own insert" on public.speak_history;
create policy "history own insert" on public.speak_history
  for insert with check (auth.uid() = user_id);
