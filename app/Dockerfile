FROM node:18-alpine AS build

WORKDIR /app

# Install deps and build
COPY app/package.json app/package-lock.json* ./
COPY app/ ./
RUN npm install --legacy-peer-deps
RUN npm run build

FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
