# The demo page, served by nginx with the agent proxied behind it. Build context: the package root.
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build:demo

FROM nginx:1.27-alpine
# nginx's image renders /etc/nginx/templates/*.template with envsubst at start: PORT and AGENT_URL.
COPY demo/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist-site /usr/share/nginx/html
