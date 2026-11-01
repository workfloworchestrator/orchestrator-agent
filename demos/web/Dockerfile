# The demo page, served by nginx with the agent proxied behind it.
FROM node:24-alpine AS build
# The chat popup is installed from its git repository, where its prepare script builds it.
RUN apk add --no-cache git
WORKDIR /app
COPY package.json ./
RUN npm install --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# nginx's image renders /etc/nginx/templates/*.template with envsubst at start: PORT and AGENT_URL.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
